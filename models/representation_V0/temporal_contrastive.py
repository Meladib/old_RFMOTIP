
import torch
import torch.nn.functional as F


class TemporalContrastiveRegularizer:

    def __init__(
        self,
        weight: float,
        temperature: float,
        pos_mode: str = "t-1",
        window: int = 5,
        neg_per_anchor: int = 64,
        neg_source: str = "same_t",
    ):
        self.weight = float(weight)
        self.temperature = float(temperature)
        self.pos_mode = pos_mode
        self.window = int(window)
        self.neg_per_anchor = int(neg_per_anchor)
        self.neg_source = neg_source

        assert self.pos_mode in ["t-1", "window"]
        assert self.neg_source in ["same_t", "all_t"]

    @torch.no_grad()
    def _collect_time_bank(self, feats_t, ids_t):

        bank = {}
        for idx, _id in enumerate(ids_t.tolist()):
            if _id not in bank:
                bank[_id] = []
            bank[_id].append(idx)
        return bank

    def __call__(self, features, id_labels, masks):

        device = features.device
        B, G, T, N, D = features.shape
        loss_sum = torch.tensor(0.0, device=device)
        count = 0

        # Flatten BG dimension for simplicity (keeps grouping intact)
        features = features.reshape(B * G, T, N, D)
        id_labels = id_labels.reshape(B * G, T, N)
        masks = masks.reshape(B * G, T, N)

        for bg in range(B * G):
            # Pre-normalize all features (stable)
            feats_bg = F.normalize(features[bg], dim=-1)  # [T, N, D]
            ids_bg = id_labels[bg]                        # [T, N]
            valid_bg = ~masks[bg]                         # [T, N]

            for t in range(1, T):
                # ---- anchors at time t ----
                a_valid = valid_bg[t]
                if a_valid.sum() == 0:
                    continue

                a_feats = feats_bg[t][a_valid]            # [A, D]
                a_ids = ids_bg[t][a_valid]                # [A]

                # ---- positives from previous time or window ----
                if self.pos_mode == "t-1":
                    pos_times = [t - 1]
                else:
                    t0 = max(0, t - self.window)
                    pos_times = list(range(t0, t))

                # Build a pool of candidate positives
                pos_feats_list = []
                pos_ids_list = []
                for tp in pos_times:
                    v = valid_bg[tp]
                    if v.sum() == 0:
                        continue
                    pos_feats_list.append(feats_bg[tp][v])
                    pos_ids_list.append(ids_bg[tp][v])

                if not pos_feats_list:
                    continue

                p_feats = torch.cat(pos_feats_list, dim=0)  # [P, D]
                p_ids = torch.cat(pos_ids_list, dim=0)      # [P]

                # For each anchor, choose one positive example with same ID
                # If none exists, skip that anchor.
                # This avoids undefined / noisy positives.
                # Also avoids any reshaping assumptions.
                pos_index_for_anchor = []
                for i in range(a_ids.numel()):
                    same = (p_ids == a_ids[i]).nonzero(as_tuple=False).flatten()
                    if same.numel() == 0:
                        pos_index_for_anchor.append(-1)
                    else:
                        # choose one positive (random) for diversity
                        j = same[torch.randint(0, same.numel(), (1,), device=device)].item()
                        pos_index_for_anchor.append(j)
                pos_index_for_anchor = torch.tensor(pos_index_for_anchor, device=device)

                keep = pos_index_for_anchor >= 0
                if keep.sum() == 0:
                    continue

                a_feats = a_feats[keep]
                a_ids = a_ids[keep]
                pos_index_for_anchor = pos_index_for_anchor[keep]
                pos_feats = p_feats[pos_index_for_anchor]   # [A', D]

                # ---- negatives ----
                if self.neg_source == "same_t":
                    # negatives from same time t (excluding same ID)
                    n_feats_pool = feats_bg[t][a_valid]
                    n_ids_pool = ids_bg[t][a_valid]
                else:
                    # negatives from all previous times (up to t), valid only
                    all_feats = []
                    all_ids = []
                    for tn in range(0, t + 1):
                        v = valid_bg[tn]
                        if v.sum() == 0:
                            continue
                        all_feats.append(feats_bg[tn][v])
                        all_ids.append(ids_bg[tn][v])
                    n_feats_pool = torch.cat(all_feats, dim=0)
                    n_ids_pool = torch.cat(all_ids, dim=0)

                # For each anchor sample K negatives with different ID
                K = self.neg_per_anchor
                # If pool is too small, reduce K
                K = min(K, max(1, n_feats_pool.size(0) - 1))

                # Compute logits: [A', 1+K]
                # logit_pos = dot(a, pos)/tau
                logit_pos = (a_feats * pos_feats).sum(dim=-1, keepdim=True) / self.temperature  # [A',1]

                # Sample negatives per anchor
                neg_logits_list = []
                for i in range(a_ids.numel()):
                    diff = (n_ids_pool != a_ids[i]).nonzero(as_tuple=False).flatten()
                    if diff.numel() == 0:
                        # If no negatives, skip this anchor (should be rare)
                        neg_logits_list.append(None)
                        continue
                    if diff.numel() < K:
                        choose = diff[torch.randint(0, diff.numel(), (K,), device=device)]
                    else:
                        choose = diff[torch.randperm(diff.numel(), device=device)[:K]]
                    neg_feats = n_feats_pool[choose]  # [K,D]
                    neg_logits = (a_feats[i:i+1] * neg_feats).sum(dim=-1) / self.temperature  # [K]
                    neg_logits_list.append(neg_logits)

                # Filter anchors where negatives were not possible
                good = [x is not None for x in neg_logits_list]
                if not any(good):
                    continue

                good_idx = torch.tensor(good, device=device)
                logit_pos_g = logit_pos[good_idx]
                a_count = int(good_idx.sum().item())
                neg_logits = torch.stack([x for x in neg_logits_list if x is not None], dim=0)  # [A_good,K]

                logits = torch.cat([logit_pos_g, neg_logits], dim=1)  # [A_good, 1+K]
                labels = torch.zeros((a_count,), dtype=torch.long, device=device)  # positive is index 0

                # Cross-entropy InfoNCE
                loss = F.cross_entropy(logits, labels, reduction="mean")
                loss_sum = loss_sum + loss
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)
        return self.weight * (loss_sum / count)

import torch
import torch.nn.functional as F


class TemporalContrastiveRegularizer:
    """
    Temporal-Positive Contrastive (TPC)
    Anchor at t, positive = same ID at t-1 or window,
    negatives = different IDs.
    """

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

    def __call__(self, features, id_labels, masks):
        device = features.device
        B, G, T, N, D = features.shape

        features = features.reshape(B * G, T, N, D)
        id_labels = id_labels.reshape(B * G, T, N)
        masks = masks.reshape(B * G, T, N)

        loss_sum = torch.tensor(0.0, device=device)
        count = 0

        for bg in range(B * G):
            feats = F.normalize(features[bg], dim=-1)
            ids = id_labels[bg]
            valid = ~masks[bg]

            for t in range(1, T):
                a_valid = valid[t]
                if a_valid.sum() == 0:
                    continue

                a_feats = feats[t][a_valid]
                a_ids = ids[t][a_valid]

                # ---- positives ----
                if self.pos_mode == "t-1":
                    pos_times = [t - 1]
                else:
                    pos_times = list(range(max(0, t - self.window), t))

                p_feats, p_ids = [], []
                for tp in pos_times:
                    v = valid[tp]
                    if v.sum() == 0:
                        continue
                    p_feats.append(feats[tp][v])
                    p_ids.append(ids[tp][v])

                if not p_feats:
                    continue

                p_feats = torch.cat(p_feats, dim=0)
                p_ids = torch.cat(p_ids, dim=0)

                # anchor → positive
                pos_idx = []
                for i in range(a_ids.numel()):
                    same = (p_ids == a_ids[i]).nonzero(as_tuple=False).flatten()
                    pos_idx.append(
                        same[torch.randint(0, same.numel(), (1,), device=device)].item()
                        if same.numel() > 0 else -1
                    )

                pos_idx = torch.tensor(pos_idx, device=device)
                keep = pos_idx >= 0
                if keep.sum() == 0:
                    continue

                a_feats = a_feats[keep]
                a_ids = a_ids[keep]
                pos_feats = p_feats[pos_idx[keep]]

                # ---- negatives ----
                if self.neg_source == "same_t":
                    n_feats = feats[t][a_valid]
                    n_ids = ids[t][a_valid]
                else:
                    n_feats, n_ids = [], []
                    for tt in range(t + 1):
                        v = valid[tt]
                        if v.sum() == 0:
                            continue
                        n_feats.append(feats[tt][v])
                        n_ids.append(ids[tt][v])
                    n_feats = torch.cat(n_feats, dim=0)
                    n_ids = torch.cat(n_ids, dim=0)

                K = min(self.neg_per_anchor, max(8, n_feats.size(0) // 4))

                logits_pos = (a_feats * pos_feats).sum(-1, keepdim=True) / self.temperature
                logits_neg = []

                for i in range(a_ids.numel()):
                    diff = (n_ids != a_ids[i]).nonzero(as_tuple=False).flatten()
                    if diff.numel() == 0:
                        continue
                    choose = diff[torch.randperm(diff.numel(), device=device)[:K]]
                    neg_feats = n_feats[choose]
                    logits_neg.append((a_feats[i:i+1] * neg_feats).sum(-1) / self.temperature)

                if not logits_neg:
                    continue

                logits_neg = torch.stack(logits_neg, dim=0)
                logits = torch.cat([logits_pos, logits_neg], dim=1)

                labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                loss_sum += F.cross_entropy(logits, labels)
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)

        return self.weight * (loss_sum / count)

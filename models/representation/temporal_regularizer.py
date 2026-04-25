import torch


class TemporalRegularizer:
    """
    ID-aware temporal smoothness:
    Enforces smooth embeddings only when identity is consistent across frames.
    """

    def __init__(self, weight: float):
        self.weight = float(weight)

    def __call__(self, features, ids, masks):
        """
        features: [B, G, T, N, D]
        ids:      [B, G, T, N]
        masks:    [B, G, T, N]  (True = invalid)
        """
        valid = ~masks

        f_t = features[:, :, 1:]
        f_tm1 = features[:, :, :-1]

        valid_pair = valid[:, :, 1:] & valid[:, :, :-1]
        same_id = ids[:, :, 1:] == ids[:, :, :-1]

        valid_pair = valid_pair & same_id

        if valid_pair.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        diff = f_t - f_tm1
        loss = (diff.pow(2).sum(-1) * valid_pair).sum()
        denom = valid_pair.sum().clamp(min=1)

        return self.weight * (loss / denom)

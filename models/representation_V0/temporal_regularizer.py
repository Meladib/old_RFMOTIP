import torch


class TemporalRegularizer:
    """
    Temporal consistency loss:
    enforces smooth embeddings for same identity over time.
    """

    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, features, masks):
        """
        features: [B, G, T, N, D]
        masks:    [B, G, T, N]  (True = invalid)
        """
        valid = ~masks

        f_t = features[:, :, 1:, :, :]
        f_tm1 = features[:, :, :-1, :, :]
        valid_pair = valid[:, :, 1:, :] & valid[:, :, :-1, :]

        f_t_n = torch.nn.functional.normalize(f_t, dim=-1)
        f_tm1_n = torch.nn.functional.normalize(f_tm1, dim=-1)

        cos_loss = (1 - (f_t_n * f_tm1_n).sum(-1)) * valid_pair
        l2_loss = (f_t - f_tm1).pow(2).sum(-1) * valid_pair

        loss = cos_loss.mean() + 0.1 * l2_loss.mean()
        return self.weight * loss

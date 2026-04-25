import torch
import torch.nn.functional as F


class MetricRegularizer:
    def __init__(self, weight: float, temperature: float):
        self.weight = weight
        self.temperature = temperature

    def __call__(self, features, id_labels, masks):
        B, G, T, N, D = features.shape
        total_loss = 0.0
        count = 0

        for t in range(T):
            f = features[:, :, t].reshape(-1, D)
            ids = id_labels[:, :, t].reshape(-1)
            valid = ~masks[:, :, t].reshape(-1)

            f = f[valid]
            ids = ids[valid]

            if f.size(0) < 2:
                continue

            f = torch.nn.functional.normalize(f, dim=-1)
            sim = torch.matmul(f, f.t()) / self.temperature

            for i in range(f.size(0)):
                pos_idx = (ids == ids[i])
                neg_idx = ~pos_idx

                # remove self-comparison
                pos_idx[i] = False

                if pos_idx.sum() == 0 or neg_idx.sum() == 0:
                    continue

                pos_sim = sim[i, pos_idx].mean()
                neg_sim = sim[i, neg_idx].mean()

                margin = 0.2
                total_loss += torch.relu(margin - pos_sim + neg_sim)
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=features.device)

        return self.weight * (total_loss / count)

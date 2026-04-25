# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch.nn as nn

from models.ffn import FFN


class TrajectoryModeling(nn.Module):
    def __init__(
            self,
            detr_dim: int,
            ffn_dim_ratio: int,
            feature_dim: int,
    ):
        super().__init__()

        self.detr_dim = detr_dim
        self.ffn_dim_ratio = ffn_dim_ratio
        self.feature_dim = feature_dim

        self.adapter = FFN(
            d_model=detr_dim,
            d_ffn=detr_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = FFN(
            d_model=feature_dim,
            d_ffn=feature_dim * ffn_dim_ratio,
            activation=nn.GELU(),
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        pass

    def forward(self, seq_info):
        for key in ("trajectory_features", "unknown_features"):
            if key not in seq_info:
                continue
            features = seq_info[key]
            features = features + self.adapter(features)
            features = self.norm(features)
            features = features + self.ffn(features)
            features = self.ffn_norm(features)
            seq_info[key] = features
        return seq_info
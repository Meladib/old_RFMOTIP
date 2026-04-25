import torch.nn as nn

from .temporal_regularizer import TemporalRegularizer
from .metric_regularizer import MetricRegularizer
from .temporal_contrastive import TemporalContrastiveRegularizer

class IdentityRepresentation(nn.Module):
    """
    Training-only identity representation stabilization module.
    Operates directly on RF-DETR embeddings.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.use_temporal = config.get("REP_USE_TEMPORAL", False)
        self.use_contrastive = config.get("REP_USE_CONTRASTIVE", False)

        self.temporal_reg = (
            TemporalRegularizer(config["REP_TEMPORAL_WEIGHT"])
            if self.use_temporal else None
        )

        self.metric_reg = (
            MetricRegularizer(
                config["REP_CONTRASTIVE_WEIGHT"],
                config["REP_CONTRASTIVE_TEMPERATURE"],
            )
            if self.use_contrastive else None
        )
        self.use_tpc = config.get("REP_USE_TPC", False)
        self.tpc_reg = (
            TemporalContrastiveRegularizer(
                weight=config["REP_TPC_WEIGHT"],
                temperature=config["REP_TPC_TEMPERATURE"],
                pos_mode=config.get("REP_TPC_POS_MODE", "t-1"),
                window=config.get("REP_TPC_WINDOW", 5),
                neg_per_anchor=config.get("REP_TPC_NEG_PER_ANCHOR", 64),
                neg_source=config.get("REP_TPC_NEG_SOURCE", "same_t"),
            )
            if self.use_tpc else None
        )

    def forward(self, seq_info: dict):
        losses = {}

        traj_feat = seq_info["trajectory_features"]
        traj_masks = seq_info["trajectory_masks"]
        traj_ids = seq_info["trajectory_id_labels"]

        if self.temporal_reg is not None and self.training:
            losses["rep_temporal"] = self.temporal_reg(
                traj_feat, traj_masks
            )

        if self.metric_reg is not None and self.training:
            losses["rep_contrastive"] = self.metric_reg(
                traj_feat, traj_ids, traj_masks
            )
        if self.tpc_reg is not None and self.training:
            losses["rep_tpc"] = self.tpc_reg(traj_feat, traj_ids, traj_masks)
        return {
            "seq_info": seq_info,   # untouched structure
            "losses": losses,
        }

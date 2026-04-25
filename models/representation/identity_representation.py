import torch.nn as nn

from .temporal_regularizer import TemporalRegularizer
from .metric_regularizer import MetricRegularizer
from .temporal_contrastive import TemporalContrastiveRegularizer


class IdentityRepresentation(nn.Module):
    """
    Training-only identity representation stabilization module.
    Operates on trajectory-aligned RF-DETR embeddings.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.use_temporal = config.get("REP_USE_TEMPORAL", False)
        self.use_contrastive = config.get("REP_USE_CONTRASTIVE", False)
        self.use_tpc = config.get("REP_USE_TPC", False)

        self.temporal_reg = (
            TemporalRegularizer(weight=config["REP_TEMPORAL_WEIGHT"])
            if self.use_temporal else None
        )

        self.metric_reg = (
            MetricRegularizer(
                weight=config["REP_CONTRASTIVE_WEIGHT"],
                temperature=config["REP_CONTRASTIVE_TEMPERATURE"],
            )
            if self.use_contrastive else None
        )

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
        """
        Returns:
            {
                "seq_info": untouched seq_info,
                "losses": dict of representation losses
            }
        """
        losses = {}

        traj_feat = seq_info["trajectory_features"]
        traj_masks = seq_info["trajectory_masks"]
        traj_ids = seq_info["trajectory_id_labels"]

        if not self.training:
            return {"seq_info": seq_info, "losses": losses}

        # ---- Temporal smoothness (ID-aware inside) ----
        if self.temporal_reg is not None:
            losses["rep_temporal"] = self.temporal_reg(
                traj_feat, traj_ids, traj_masks
            )

        # ---- Frame-wise metric regularization (optional) ----
        if self.metric_reg is not None:
            losses["rep_contrastive"] = self.metric_reg(
                traj_feat, traj_ids, traj_masks
            )

        # ---- Temporal-positive contrastive (TPC) with curriculum ----
        if self.tpc_reg is not None:
            base_loss = self.tpc_reg(traj_feat, traj_ids, traj_masks)

            # curriculum decay (strong early, weak later)
            step = seq_info.get("global_step", None)
            if step is not None:
                decay = max(0.2, 1.0 - step / 20000)
                losses["rep_tpc"] = base_loss * decay
            else:
                losses["rep_tpc"] = base_loss

        return {
            "seq_info": seq_info,
            "losses": losses,
        }

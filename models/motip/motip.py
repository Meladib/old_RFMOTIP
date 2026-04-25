# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch

class MOTIP(nn.Module):
    def __init__(
            self,
            detr: nn.Module,
            detr_framework: str,
            only_detr: bool,
            trajectory_modeling: nn.Module,
            id_decoder: nn.Module,
            identity_representation: nn.Module | None = None,   # NEW
    ):
        super().__init__()
        self.detr = detr
        self.detr_framework = detr_framework
        self.only_detr = only_detr
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder
        self.identity_representation = identity_representation

        if self.id_decoder is not None:
            self.num_id_vocabulary = self.id_decoder.num_id_vocabulary
        else:
            self.num_id_vocabulary = 1000           # hack implementation

        return
    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for MOTIP forward."
        match kwargs["part"]:
            case "detr":
                frames = kwargs["frames"]

                # 🔒 HARD LOCK RF-DETR TO EVAL
                self.detr.eval()
                with torch.no_grad():
                    detr_out = self.detr(samples=frames)
                # ---------------- SAFETY ASSERT ----------------
                Q = detr_out["pred_logits"].shape[1]
                expected_Q = self.detr.num_queries
                assert Q == expected_Q, (
                    f"[FATAL] RF-DETR returned {Q} queries, expected {expected_Q}. "
                    f"RF-DETR is NOT frozen correctly."
                )
                return detr_out

            case "representation":   # NEW
                assert self.identity_representation is not None, \
                    "identity_representation is not built but 'representation' was called"
                return self.identity_representation(kwargs["seq_info"])            

            case "trajectory_modeling":
                return self.trajectory_modeling(kwargs["seq_info"])

            case "id_decoder":
                return self.id_decoder(
                    kwargs["seq_info"],
                    use_decoder_checkpoint=kwargs.get("use_decoder_checkpoint", False)
                )

            case _:
                raise NotImplementedError

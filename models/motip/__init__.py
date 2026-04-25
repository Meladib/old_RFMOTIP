# Copyright (c) Ruopeng Gao. All Rights Reserved.
import torch
import argparse

from models.rf_detr.lwdetr import (
    build_model,
    build_motip_criterion,
    PostProcess
)

from models.rf_detr.util.utils import clean_state_dict
from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder
#from models.representation.identity_representation import IdentityRepresentation
from models.motip.motip import MOTIP



torch.serialization.add_safe_globals([argparse.Namespace])

def build(config: dict):

    ckpt_path = config["CKPT_PATH"]
    print(f"\n================= MOTIP RF-DETR LOADING =================")
    print(f"Checkpoint: {ckpt_path}")

    # -------------------------------------------------------
    # 1) LOAD CHECKPOINT (same as inference script)
    # -------------------------------------------------------
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args_ckpt = ckpt["args"]
    state_dict = clean_state_dict(ckpt["model"])

    detr_model = build_model(args_ckpt)     # <== identical to working inference
    postprocess = PostProcess(num_select=args_ckpt.num_select)

    print(">>> Reinitializing detection head → 1 class")
    detr_model.reinitialize_detection_head(1)

    missing, unexpected = detr_model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detr_model = detr_model.to(device).eval()

    # This is the REAL DETR module (LWDETR)
    real_detr = detr_model
    # ================================================================
    # =====================   DEBUG SECTION   ========================
    # ================================================================
    if config.get("DEBUG_MOTIP", False):
        print("\n================= DEBUG: RF-DETR SANITY CHECK =================")


        dummy = torch.randn(1, 3, 320, 320).to(device)
        dummy_mask = torch.zeros(1, 320, 320, dtype=torch.bool).to(device)
        from models.rf_detr.util.misc import NestedTensor
        nt = NestedTensor(dummy, dummy_mask)

        with torch.no_grad():
            out = real_detr(nt)

        print("\n--- DETR output keys:", out.keys())
        print("pred_logits:", out["pred_logits"].shape)
        print("pred_boxes:", out["pred_boxes"].shape)
        print("outputs (hs_last):", out["outputs"].shape)

        # Check typical expected MOTIP shapes
        expected_queries = args_ckpt.num_queries
        print("\nExpected queries =", expected_queries)

        if out["pred_logits"].shape[1] != expected_queries:
            print("❌ ERROR: RF-DETR inference query count != MOTIP expected query count")

        # Check feature distribution
        feats = out["outputs"]  # shape [1, num_queries, hidden_dim]
        print("\nFeature stats:")
        print("  mean:", feats.mean().item())
        print("  std:", feats.std().item())
        print("  max:", feats.max().item(), "min:", feats.min().item())

        # Check if any query is frozen or dead (common bug)
        dead_queries = (feats.norm(dim=-1) < 1e-6).sum().item()
        print("Dead queries (norm < 1e-6):", dead_queries)

        print("===============================================================\n")


    print(">>> RF-DETR successfully loaded for MOTIP.")

    # -------------------------------------------------------
    # 3) Build MOTIP-side modules (trajectory + ID decoder)
    # -------------------------------------------------------


    if config["ONLY_DETR"]:
        trajectory_modeling = None
        id_decoder = None
    else:
        trajectory_modeling = TrajectoryModeling(
            detr_dim=config["HIDDEN_DIM"],
            ffn_dim_ratio=config["FFN_DIM_RATIO"],
            feature_dim=config["FEATURE_DIM"],
        )

        id_decoder = IDDecoder(
            feature_dim=config["FEATURE_DIM"],
            id_dim=config["ID_DIM"],
            ffn_dim_ratio=config["FFN_DIM_RATIO"],
            num_layers=config["NUM_ID_DECODER_LAYERS"],
            head_dim=config["HEAD_DIM"],
            num_id_vocabulary=config["NUM_ID_VOCABULARY"],
            rel_pe_length=config["REL_PE_LENGTH"],
            use_aux_loss=config["USE_AUX_LOSS"],
            use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
        )
        identity_representation = None
        if config.get("USE_IDENTITY_REPRESENTATION", False):
            from models.representation.identity_representation import IdentityRepresentation
            identity_representation = IdentityRepresentation(config=config)

    # -------------------------------------------------------
    # 4) Build MOTIP model using REAL detr_model
    # -------------------------------------------------------
    motip_model = MOTIP(
        detr=real_detr,     # IMPORTANT — now the same RF-DETR used in inference
        detr_framework="rf_detr",
        only_detr=config["ONLY_DETR"],
        trajectory_modeling=trajectory_modeling,
        id_decoder=id_decoder,
        identity_representation=identity_representation,  # NEW
    )

    print(">>> MOTIP successfully built.")

    # -------------------------------------------------------
    # 5) Build MOTIP Criterion using args_ckpt
    # -------------------------------------------------------
    args_ckpt.device = device.type
    criterion, _ = build_motip_criterion(args_ckpt)

    print(">>> MOTIP criterion built.\n")



    return motip_model, criterion

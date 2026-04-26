"""
Temporal Stability Diagnostic Script for RF-MOTIP.

Measures cross-frame cosine similarity of RF-DETR query embeddings across
consecutive frame pairs. Uses a PyTorch forward hook on the TransformerDecoder
to capture intermediate layer outputs (not just the final layer).

Usage:
    python diagnostics/diag_temporal_stability_script.py \
        --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
        --sequence_dir /data/DanceTrack/val/dancetrack0001 \
        --layer_index 2 \
        --output_dir diagnostics/
"""

import argparse
import json
import os
import sys
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import v2

# Ensure repo root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="RF-MOTIP Temporal Stability Diagnostic")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to RF-DETR checkpoint (.pth), with 'model' and 'args' keys.")
    parser.add_argument("--sequence_dir", required=True,
                        help="Path to a DanceTrack sequence directory (contains img1/).")
    parser.add_argument("--deformable_checkpoint", default=None,
                        help="Optional: path to a deformable DETR checkpoint (unused, for compat).")
    parser.add_argument("--layer_index", type=int, default=-1,
                        help="Decoder layer index to extract embeddings from. "
                             "0=first, 1=middle, 2=final (default: -1 → final layer).")
    parser.add_argument("--output_dir", default="diagnostics/",
                        help="Directory to write temporal_stability_results.json.")
    parser.add_argument("--num_pairs", type=int, default=5,
                        help="Number of consecutive frame pairs to evaluate.")
    parser.add_argument("--det_thresh", type=float, default=0.3,
                        help="Detection score threshold for filtering active queries.")
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load RF-DETR model from a MOTIP-style checkpoint."""
    import argparse as _argparse
    torch.serialization.add_safe_globals([_argparse.Namespace])

    from models.rf_detr.lwdetr import build_model
    from models.rf_detr.util.utils import clean_state_dict

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args_ckpt = ckpt["args"]
    state_dict = clean_state_dict(ckpt["model"])

    model = build_model(args_ckpt)
    model.reinitialize_detection_head(1)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, args_ckpt


def load_frame(image_path, device, target_size=512):
    """Load and preprocess a single frame into a NestedTensor."""
    from models.rf_detr.util.misc import NestedTensor

    img = Image.open(image_path).convert("RGB")
    # Resize so the longer side is target_size
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Convert to tensor and normalize
    tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.to(device)

    mask = torch.zeros(1, new_h, new_w, dtype=torch.bool, device=device)
    return NestedTensor(tensor.unsqueeze(0), mask)


def get_frame_paths(sequence_dir, num_pairs):
    """Return a list of frame image paths for the first num_pairs+1 frames."""
    img_dir = os.path.join(sequence_dir, "img1")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    frames = sorted(
        [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))],
    )
    if len(frames) < num_pairs + 1:
        raise ValueError(
            f"Sequence has only {len(frames)} frames, need at least {num_pairs + 1}."
        )
    return [os.path.join(img_dir, f) for f in frames[:num_pairs + 1]]


def extract_embeddings(model, nested_tensor, layer_index, det_thresh):
    """
    Run LWDETR forward and extract embeddings at the specified decoder layer.

    Uses a forward hook on the TransformerDecoder to capture the stacked
    intermediate output tensor hs of shape [num_layers, B, N, C].

    Returns:
        embeddings: Tensor [N_active, C] — embeddings of score-filtered queries
        scores: Tensor [N_active] — detection scores of active queries
    """
    captured = {}

    def _hook(module, input, output):
        # output is list: [hs_stacked, refs_stacked]
        # hs_stacked shape: [num_layers, B, N, C]
        if isinstance(output, list) and len(output) >= 1 and isinstance(output[0], torch.Tensor):
            captured["hs"] = output[0].detach()  # [num_dec_layers, B, N, C]

    hook = model.transformer.decoder.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            detr_out = model(nested_tensor)
    finally:
        hook.remove()

    if "hs" not in captured:
        raise RuntimeError(
            "TransformerDecoder hook did not fire. "
            "Check that the model uses TransformerDecoder with return_intermediate=True."
        )

    hs = captured["hs"]  # [num_layers, B, N, C]
    num_layers = hs.shape[0]

    # Resolve layer_index (supports negative indexing)
    layer_idx = layer_index if layer_index >= 0 else num_layers + layer_index
    if not (0 <= layer_idx < num_layers):
        raise ValueError(
            f"layer_index {layer_index} resolves to {layer_idx}, "
            f"but decoder has {num_layers} layers (0 to {num_layers - 1})."
        )

    embeddings_all = hs[layer_idx, 0]  # [N, C]

    # Use pred_logits from the final layer to get detection scores
    logits = detr_out["pred_logits"][0]  # [N, num_classes]
    scores, _ = torch.max(logits.sigmoid(), dim=-1)  # [N]

    active_mask = scores > det_thresh
    return embeddings_all[active_mask], scores[active_mask]


def cosine_similarities_matched(emb_t, emb_t1):
    """
    For each query in emb_t, find the nearest-neighbor in emb_t1 by cosine
    similarity, and return those per-query max cosine similarities.

    Args:
        emb_t:  [N_t, C]
        emb_t1: [N_t1, C]

    Returns:
        cos_sims: [N_t] — cosine similarity to best match in t+1
    """
    emb_t_norm = F.normalize(emb_t, dim=-1)
    emb_t1_norm = F.normalize(emb_t1, dim=-1)
    # [N_t, N_t1] similarity matrix
    sim_matrix = torch.mm(emb_t_norm, emb_t1_norm.t())
    # Best match for each query in t
    cos_sims, _ = sim_matrix.max(dim=-1)
    return cos_sims


def compute_histogram(values, n_bins=10):
    """Compute a histogram over [-1, 1]."""
    counts, bin_edges = np.histogram(values, bins=n_bins, range=(-1.0, 1.0))
    return {
        "bins": [(float(bin_edges[i]), float(bin_edges[i + 1])) for i in range(n_bins)],
        "counts": counts.tolist(),
    }


def print_summary_table(pairs_data, summary):
    width = 70
    print("\n" + "=" * width)
    print("RF-MOTIP TEMPORAL STABILITY DIAGNOSTIC".center(width))
    print("=" * width)
    print(f"{'Pair':<6} {'T→T+1':<12} {'N':<6} {'Mean':>7} {'Std':>7} "
          f"{'Min':>7} {'Max':>7} {'Unstable':>9} {'Stable':>7}")
    print("-" * width)
    for p in pairs_data:
        print(
            f"{p['pair_id']:<6} "
            f"{p['frame_t']:04d}→{p['frame_t1']:04d}   "
            f"{p['n_matched']:<6} "
            f"{p['mean_cos_sim']:>7.4f} "
            f"{p['std_cos_sim']:>7.4f} "
            f"{p['min_cos_sim']:>7.4f} "
            f"{p['max_cos_sim']:>7.4f} "
            f"{p['n_unstable']:>9} "
            f"{p['n_stable']:>7}"
        )
    print("=" * width)
    print(f"{'SUMMARY':<6} {'':12} "
          f"{summary['n_matched_total']:<6} "
          f"{summary['mean_cos_sim']:>7.4f} "
          f"{summary['std_cos_sim']:>7.4f} "
          f"{summary['min_cos_sim']:>7.4f} "
          f"{summary['max_cos_sim']:>7.4f} "
          f"{summary['n_unstable_total']:>9} "
          f"{summary['n_stable_total']:>7}")
    print("=" * width)

    # Interpretation
    m = summary["mean_cos_sim"]
    if m > 0.95:
        level = "STABLE"
    elif m > 0.85:
        level = "ACCEPTABLE"
    elif m > 0.70:
        level = "DEGRADED"
    elif m > 0.50:
        level = "SEVERE"
    else:
        level = "CRITICAL"

    unstable_frac = summary["n_unstable_total"] / max(summary["n_matched_total"], 1)
    print(f"\n  Stability level : {level}  (mean={m:.4f})")
    print(f"  Unstable queries: {unstable_frac * 100:.1f}%  "
          f"({summary['n_unstable_total']} / {summary['n_matched_total']})")
    print("=" * width + "\n")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sequence: {args.sequence_dir}")
    print(f"Layer index: {args.layer_index}  (−1 = final layer)")

    # Load model
    print("\nLoading model...")
    model, args_ckpt = load_model(args.checkpoint, device)
    print(f"  dec_layers: {getattr(args_ckpt, 'dec_layers', '?')}")
    print(f"  hidden_dim: {getattr(args_ckpt, 'hidden_dim', '?')}")
    print(f"  num_queries: {getattr(args_ckpt, 'num_queries', '?')}")

    # Load frame paths
    frame_paths = get_frame_paths(args.sequence_dir, args.num_pairs)
    print(f"\nFound {len(frame_paths)} frames; evaluating {args.num_pairs} pairs.")

    # Evaluate pairs
    pairs_data = []
    all_cos_sims = []

    for pair_id in range(args.num_pairs):
        path_t = frame_paths[pair_id]
        path_t1 = frame_paths[pair_id + 1]

        nt_t = load_frame(path_t, device)
        nt_t1 = load_frame(path_t1, device)

        emb_t, scores_t = extract_embeddings(model, nt_t, args.layer_index, args.det_thresh)
        emb_t1, scores_t1 = extract_embeddings(model, nt_t1, args.layer_index, args.det_thresh)

        if emb_t.shape[0] == 0 or emb_t1.shape[0] == 0:
            print(f"  Pair {pair_id}: no active detections (thresh={args.det_thresh}), skipping.")
            continue

        cos_sims = cosine_similarities_matched(emb_t, emb_t1).cpu().numpy()
        all_cos_sims.extend(cos_sims.tolist())

        n_unstable = int((cos_sims < 0.5).sum())
        n_stable = int((cos_sims > 0.9).sum())
        hist = compute_histogram(cos_sims)

        # Parse frame indices from filename
        frame_t_idx = int(os.path.splitext(os.path.basename(path_t))[0])
        frame_t1_idx = int(os.path.splitext(os.path.basename(path_t1))[0])

        pair_result = {
            "pair_id": pair_id,
            "frame_t": frame_t_idx,
            "frame_t1": frame_t1_idx,
            "n_matched": len(cos_sims),
            "mean_cos_sim": float(np.mean(cos_sims)),
            "std_cos_sim": float(np.std(cos_sims)),
            "min_cos_sim": float(np.min(cos_sims)),
            "max_cos_sim": float(np.max(cos_sims)),
            "n_unstable": n_unstable,
            "n_stable": n_stable,
            "histogram": hist,
        }
        pairs_data.append(pair_result)

    if not pairs_data:
        print("ERROR: No valid frame pairs found. Lower --det_thresh or check sequence.")
        sys.exit(1)

    # Summary
    all_cos_sims_arr = np.array(all_cos_sims)
    summary = {
        "mean_cos_sim": float(np.mean(all_cos_sims_arr)),
        "std_cos_sim": float(np.std(all_cos_sims_arr)),
        "min_cos_sim": float(np.min(all_cos_sims_arr)),
        "max_cos_sim": float(np.max(all_cos_sims_arr)),
        "n_unstable_total": int((all_cos_sims_arr < 0.5).sum()),
        "n_stable_total": int((all_cos_sims_arr > 0.9).sum()),
        "n_matched_total": len(all_cos_sims_arr),
    }

    results = {
        "layer_index": args.layer_index,
        "checkpoint": args.checkpoint,
        "sequence_dir": args.sequence_dir,
        "det_thresh": args.det_thresh,
        "pairs": pairs_data,
        "summary": summary,
    }

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "temporal_stability_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Print table
    print_summary_table(pairs_data, summary)


if __name__ == "__main__":
    main()

# DIAG_TEMPORAL_STABILITY

## 1. WHAT THE SCRIPT MEASURES AND WHY

The `diag_temporal_stability_script.py` measures **cross-frame cosine similarity
of RF-DETR query embeddings** — specifically, for each detected object in frame T,
it finds the nearest-neighbor embedding in frame T+1 and computes
`cos_sim(e_T, e_{T+1})`.

### Why this matters

MOTIP's ID decoder maintains a trajectory dictionary where historical tracklet
tokens `τ^m_k = concat(f^m_t, i^k_m)` accumulate over time. The ID decoder
performs cross-attention between the current frame's unknown queries and historical
tracklets to predict identity labels.

For this to work, **the feature vector `f^n_t` for the same physical object must
remain stable (high cosine similarity) across consecutive frames**. If the cosine
similarity is low, the cross-attention query vector in frame T+1 will not match
the key/value vectors from frame T, and the ID decoder's attention will be
diffuse — unable to concentrate on the correct tracklet.

Three sources of instability identified by static analysis (Axes 1–4) that this
script quantifies:

1. **Encoder top-K slot re-selection** (Axis 3): different encoder slots may be
   selected for the same object per frame, producing embeddings from different
   learned weight positions.
2. **LayerNorm whitening** (Axis 2): normalizes feature magnitudes, reducing the
   "distance" between different objects and making temporal matching harder.
3. **NAS weight sharing** (Axis 1): the 3-layer decoder was trained under random
   depth configurations 1–6; no single depth position is specialized.

### The script quantifies

- Mean cosine similarity between matched detections across consecutive frame pairs
- Standard deviation (spread of stability across queries)
- Minimum (worst-case instability — the queries most likely to cause ID swaps)
- Number of "unstable" queries (cosine sim < 0.5)
- Number of "stable" queries (cosine sim > 0.9)
- Histogram distribution across 10 bins [−1.0, 1.0]
- Per-pair statistics across 5 consecutive frame pairs

---

## 2. EXACT CLI COMMANDS

### Basic usage (frozen RF-DETR checkpoint, final decoder layer)

```bash
cd /path/to/old_RFMOTIP
python diagnostics/diag_temporal_stability_script.py \
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
    --sequence_dir /data/pos+mot/Datadir/DanceTrack/val/dancetrack0001 \
    --output_dir diagnostics/
```

### Probe all three decoder layers

```bash
# Layer 0 (earliest, least processed)
python diagnostics/diag_temporal_stability_script.py \
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
    --sequence_dir /data/pos+mot/Datadir/DanceTrack/val/dancetrack0001 \
    --layer_index 0 \
    --output_dir diagnostics/

# Layer 1 (intermediate)
python diagnostics/diag_temporal_stability_script.py \
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
    --sequence_dir /data/pos+mot/Datadir/DanceTrack/val/dancetrack0001 \
    --layer_index 1 \
    --output_dir diagnostics/

# Layer 2 (final, consumed by MOTIP — default)
python diagnostics/diag_temporal_stability_script.py \
    --checkpoint rfdetr_dancetrack_motip/checkpoint_best_total.pth \
    --sequence_dir /data/pos+mot/Datadir/DanceTrack/val/dancetrack0001 \
    --layer_index 2 \
    --output_dir diagnostics/
```

### Generate plots after running the script

```bash
python diagnostics/plot_temporal_stability.py \
    --results_json diagnostics/temporal_stability_results.json \
    --output_dir diagnostics/
```

This produces:
- `diagnostics/temporal_stability_histogram.png` — overlaid histogram across pairs
- `diagnostics/temporal_stability_per_pair.png` — line chart of mean ± std per pair

---

## 3. INTERPRETATION THRESHOLDS

| Mean Cosine Similarity | Interpretation |
|---|---|
| > 0.95 | **Stable** — features are highly consistent; ID decoder can reliably match |
| 0.85 – 0.95 | **Acceptable** — some variability; ID decoder may struggle on low-confidence tracks |
| 0.70 – 0.85 | **Degraded** — significant instability; expect elevated ID switches at fast motion |
| 0.50 – 0.70 | **Severe** — embeddings poorly correlated; ID decoder operating near random |
| < 0.50 | **Critical** — embeddings actively anti-correlated or orthogonal; identity association is broken |

### n_unstable (cos_sim < 0.5)

If `n_unstable / n_total > 0.1` (more than 10% of matched queries are unstable),
the feature instability is a primary cause of ID association failures.

### Layer comparison interpretation

If layer-0 or layer-1 stability is substantially *higher* than layer-2 stability,
the multi-layer detection loss (in the hypothesized joint training) is degrading
the final-layer features more than earlier layers — consistent with the ID loss
backpropagating through layer-2 more strongly.

If all layers show similar low stability, the problem is primarily in the encoder
(top-K selection instability) rather than in the decoder layers.

---

## 4. HOW TO PROBE ALL 3 DECODER LAYERS

The script uses a PyTorch forward hook on
`model.detr.transformer.decoder` to capture the stacked intermediate tensor
`hs = [num_layers, B, N, C]` (shape `[3, 1, 300, 256]` for RF-DETR-small).
The `--layer_index` argument (0, 1, or 2) selects which layer's output to analyze.

Layer-2 (default, `--layer_index -1` or `--layer_index 2`) is the output that
MOTIP actually consumes via `out["outputs"] = hs[-1]` in `lwdetr.py:172`.

Running all three layer indices is recommended:
- Layer 0 probes encoder-dominated features (minimal decoder processing)
- Layer 2 probes the exact features consumed by the ID decoder
- Comparing layers isolates whether instability is encoder-side or decoder-side

---

## 5. OUTPUT FILE FORMAT

Results are saved to `{output_dir}/temporal_stability_results.json`:

```json
{
  "layer_index": 2,
  "checkpoint": "rfdetr_dancetrack_motip/checkpoint_best_total.pth",
  "sequence_dir": "...",
  "pairs": [
    {
      "pair_id": 0,
      "frame_t": 1,
      "frame_t1": 2,
      "n_matched": 45,
      "mean_cos_sim": 0.62,
      "std_cos_sim": 0.18,
      "min_cos_sim": 0.11,
      "max_cos_sim": 0.94,
      "n_unstable": 8,
      "n_stable": 12,
      "histogram": {"bins": [...], "counts": [...]}
    },
    ...
  ],
  "summary": {
    "mean_cos_sim": 0.65,
    "std_cos_sim": 0.20,
    "min_cos_sim": 0.11,
    "max_cos_sim": 0.94,
    "n_unstable_total": 38,
    "n_stable_total": 67,
    "n_matched_total": 225
  }
}
```

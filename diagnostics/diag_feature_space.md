# DIAG_FEATURE_SPACE

## 1. FINDINGS

- **DINOv2-S backbone produces 384-dim features.**
  `models/rf_detr/backbone/dinov2.py:20`: `size_to_width["small"] = 384`.
  The backbone hidden dimension is 384 throughout all 12 transformer layers.

- **MultiScaleProjector outputs `args.hidden_dim`-dim features.**
  `models/rf_detr/lwdetr.py:321`: `out_channels=args.hidden_dim`.
  `models/rf_detr/backbone/projector.py:224-228`: the final step of each projector
  stage is `C2f(in_dim, out_channels, ...) → get_norm('LN', out_channels)` —
  a C2f block reducing to `out_channels`, followed immediately by a `LayerNorm`.
  With `args_ckpt.hidden_dim=256` (RF-DETR-small), the projector maps 384→256
  and applies LayerNorm at the output.

- **`hs[-1]` shape at the tracklet concat: `[B, 300, 256]`.**
  `models/rf_detr/lwdetr.py:172`: `out["outputs"] = hs[-1]`.
  This is `f^n_t` in MOTIP notation; shape matches MOTIP's required 256-dim.

- **TrajectoryModeling does not change dimension.**
  `models/motip/trajectory_modeling.py:21-25`: `adapter = FFN(d_model=256, d_ffn=512)`.
  Output is 256→256 with a residual add and LayerNorm (`self.norm = LayerNorm(256)`).
  `f^n_t` remains 256-dim throughout trajectory modeling.

- **Tracklet concat is correctly 512-dim.**
  `models/motip/id_decoder.py:111-112`:
  `trajectory_embeds = torch.cat([trajectory_features, trajectory_id_embeds], dim=-1)`.
  `feature_dim=256` + `id_dim=256` = 512. Matches MOTIP's `τ = 2C = 512`.

- **IDDecoder embed_dim is 512.**
  `models/motip/id_decoder.py:33`: `n_heads = (feature_dim + id_dim) // head_dim = 512 // 32 = 16`.
  Self-attention and cross-attention both use `embed_dim=512`.

- **Runtime tracker hardcodes 256.**
  `models/runtime_tracker.py:72`: `self.trajectory_features = torch.zeros((0, 0, 256), ...)`.
  This confirms 256 is the expected feature dim at inference — consistent with
  the projector output.

- **LayerNorm in the projector suppresses magnitude information.**
  `models/rf_detr/backbone/projector.py:225-228`:
  `get_norm('LN', out_channels)` = `LayerNorm(256)` is the final operation.
  `models/rf_detr/backbone/projector.py:39-47`: LayerNorm normalizes over the
  channel dimension, centering to zero mean and scaling to unit variance.
  This whitens the feature distribution: magnitude-encoded identity signals
  (e.g., object-specific scale differences) are removed. Two different objects
  with proportional feature vectors receive the same normalized output.

- **TrajectoryModeling adds a second normalization.**
  `models/motip/trajectory_modeling.py:26`: `self.norm = nn.LayerNorm(feature_dim)`.
  Applied after the residual adapter. Features are normalized twice before
  reaching the ID decoder: once in the projector, once in trajectory modeling.
  This double whitening further suppresses any amplitude-based identity signal
  that may have survived the projector.

---

## 2. MISMATCH EVIDENCE

| Location | Observed | Expected | Impact |
|---|---|---|---|
| `backbone/projector.py:225-228` | LayerNorm as final output step | Unnormalized or BN-normalized | All magnitude-encoded identity information is whitened before reaching ID decoder |
| `motip/trajectory_modeling.py:26` | Second LayerNorm applied to features | Single normalization | Compound whitening; identity signals are further compressed |
| `dinov2.py:20` | 384-dim backbone → projected to 256 | MOTIP expects native 256 | Projection is correct in shape but introduces a learned dimensionality reduction |
| `runtime_tracker.py:72` | Hardcoded `256` in trajectory buffer init | Should be `feature_dim` from config | Breaks if `args_ckpt.hidden_dim ≠ 256` at runtime |

---

## 3. DIMENSION TABLE

| Module | Input Shape | Output Shape | Consumed By |
|---|---|---|---|
| DINOv2-S backbone | `[B, 3, H, W]` | `[B, 384, H/14, W/14]` per layer | MultiScaleProjector |
| MultiScaleProjector (C2f) | `[B, 384, h, w]` multi-scale | `[B, 256, h', w']` multi-scale | TransformerDecoder memory |
| LayerNorm in Projector | `[B, 256, h', w']` | `[B, 256, h', w']` normalized | TransformerDecoder memory |
| TransformerDecoder | `[B, 300, 256]` queries + memory | `[3, B, 300, 256]` hs stack | LWDETR.forward |
| `hs[-1]` extraction | `[3, B, 300, 256]` | `[B, 300, 256]` | prepare_for_motip / runtime tracker |
| TrajectoryModeling (adapter) | `[B, G, T, N, 256]` | `[B, G, T, N, 256]` | IDDecoder |
| LayerNorm in TrajectoryModeling | `[B, G, T, N, 256]` | `[B, G, T, N, 256]` normalized | IDDecoder |
| Tracklet concat | `[..., 256]` feat + `[..., 256]` id | `[..., 512]` tracklet | IDDecoder cross-attention |
| IDDecoder embed_to_word | `[..., 256]` (id half) | `[..., 51]` logits | IDCriterion |

---

## 4. SEVERITY ASSESSMENT

**High**

The double LayerNorm pipeline (projector output + trajectory modeling) whitens
identity-discriminative amplitude information, structurally explaining why
AssA=20.5 is low even when DetA=72.9 is high in the frozen experiment — the
features detect objects reliably but carry insufficient identity information for
cross-frame association, consistent with the HOTA drop from ~70 to 38.5.

---

## 5. OPEN QUESTIONS

1. **Is `args_ckpt.hidden_dim` exactly 256?**
   The YAML says 256 and RF-DETR-small paper confirms 256, but the actual
   checkpoint `args` object must be read to verify. A mismatch would cause a
   shape error at the tracklet concat.

2. **Are DINOv2 features identity-discriminative before projection?**
   DINOv2-S features are trained on image-level tasks; whether they encode
   per-instance identity across video frames is not established. The LayerNorm
   whitening may be destroying what little identity signal exists.

3. **Does the C2f dimensionality reduction (384→256) destroy identity-specific
   directions?**
   The projection is trained jointly with detection objectives, not association.
   The learned projection may map identity-discriminative 384-dim directions onto
   the null space of the 256-dim output.

---

## 5. HYPOTHESIS RANKING

#1 [Confidence: High] — The double LayerNorm normalization (projector + trajectory
modeling) whitens amplitude-encoded identity signals before they reach the ID
decoder, causing the ID decoder to receive features that are geometrically similar
for different object instances, explaining AssA=20.5 with DetA=72.9 in the frozen
experiment.

#2 [Confidence: Medium] — The C2f projection (384→256) discards 128 dimensions
that may contain identity-discriminative DINOv2 representations; the retained 256
dimensions are selected to minimize detection loss, not identity separability.

#3 [Confidence: Low] — `runtime_tracker.py:72` hardcodes 256-dim trajectory
buffers; if `args_ckpt.hidden_dim` ever differs from 256, trajectory features
will be allocated with the wrong size and silently corrupt the ID decoder inputs.

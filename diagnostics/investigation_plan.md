# Phase 1 Investigation Plan

## Context
The task is to produce `diagnostics/investigation_plan.md` — a structured map of the
RF-MOTIP codebase that locates exact file:line targets for five diagnostic axes.
No source files are modified; no diagnostic reports are written yet. This plan
describes what the output file will contain and the single commit/push needed.

## Key Findings from Exploration

### Actual repo layout (paths differ from CLAUDE.md)
- CLAUDE.md says `rfdetr/` and `motip/` but the real paths are:
  - `models/rf_detr/`  (RF-DETR source)
  - `models/motip/`    (MOTIP source)
- `diagnostics/` directory does NOT exist yet — must be created

### File inventory
```
models/rf_detr/
  lwdetr.py               Main model class LWDETR
  transformer.py          TransformerDecoder (decoder + encoder)
  backbone/
    backbone.py           Backbone wrapper
    dinov2.py             DinoV2 class — DINOv2-S hidden_size=384
    projector.py          MultiScaleProjector — out_channels=args.hidden_dim
  matcher.py              HungarianMatcher
  ops/modules/ms_deform_attn.py

models/motip/
  __init__.py             PRIMARY integration file (build_motip)
  motip.py                MOTIP wrapper class
  id_decoder.py           IDDecoder — expects feature_dim=256, id_dim=256
  trajectory_modeling.py  TrajectoryModeling — adapter FFN
  id_criterion.py         IDCriterion — cross-entropy or focal

configs/rf_motip_DT_V0motip.yaml   Configuration
train.py                           Training loop
submit_and_evaluate.py             Inference
models/runtime_tracker.py         Runtime inference tracker
models/misc.py                    load_checkpoint()
```

---

## The diagnostics/investigation_plan.md to write

```markdown
# Investigation Plan

## Repo Map

```
models/rf_detr/                     ← RF-DETR source
  lwdetr.py                         ← LWDETR main model (forward → pred_logits, pred_boxes, outputs, aux_outputs)
  transformer.py                    ← TransformerDecoder + TransformerEncoder
  matcher.py                        ← HungarianMatcher
  backbone/
    backbone.py                     ← Backbone wrapper
    dinov2.py                       ← DinoV2 (hidden_size=384 for small)
    projector.py                    ← MultiScaleProjector (out_channels=hidden_dim)
    dinov2_configs/
      dinov2_small.json             ← "hidden_size": 384

models/motip/                       ← MOTIP source
  __init__.py                       ← build_motip() — PRIMARY integration file
  motip.py                          ← MOTIP wrapper, routes by part= argument
  id_decoder.py                     ← IDDecoder (feature_dim+id_dim=512 tracklet concat)
  trajectory_modeling.py            ← TrajectoryModeling (adapter FFN, no dim change)
  id_criterion.py                   ← IDCriterion (cross-entropy over K+1=51 classes)

configs/rf_motip_DT_V0motip.yaml   ← Runtime config
train.py                            ← Training loop (prepare_for_motip at line 712)
submit_and_evaluate.py             ← Inference entry point
models/runtime_tracker.py          ← Online tracker (inference)
models/misc.py                     ← load_checkpoint()
```

## Axis 1 — Decoder Topology

- Target files:
  - `models/rf_detr/lwdetr.py:114` — `LWDETR.forward(samples, targets=None)`
  - `models/rf_detr/lwdetr.py:150-176` — return dict construction
  - `models/rf_detr/lwdetr.py:172` — `out["outputs"] = hs[-1]` (f^n_t source)
  - `models/rf_detr/lwdetr.py:210-213` — `_set_aux_loss()` builds aux_outputs list
  - `models/rf_detr/lwdetr.py:231-244` — `DummyCriterion.forward()` returns `{}, indices`
  - `models/rf_detr/transformer.py:306` — `TransformerDecoder` class definition
  - `models/rf_detr/transformer.py:343` — TransformerDecoder.forward() signature
  - `models/rf_detr/transformer.py:156` — `return_intermediate=True` flag
  - `configs/rf_motip_DT_V0motip.yaml:67` — `DETR_AUX_LOSS: True`
  - `configs/rf_motip_DT_V0motip.yaml:75` — `DETR_DEC_LAYERS: 6`
  - `models/motip/__init__.py:34` — `detr_model = build_model(args_ckpt)` (not from YAML)
  - `models/motip/__init__.py:38` — `self.detr.eval()` (frozen)
  - `train.py:412` — `detr_criterion(outputs=detr_outputs, targets=detr_targets_flatten, ...)`
  - `train.py:474-490` — loss composition (DETR loss + ID loss)

- Key question: Does `DETR_AUX_LOSS: True` in the YAML actually activate per-layer detection
  losses during joint training, and does `args_ckpt.dec_layers` (3, from checkpoint) conflict
  with `DETR_DEC_LAYERS: 6` in the YAML?

- Blocker: The exact value of `args_ckpt.dec_layers` requires reading the saved checkpoint.
  The YAML says 6 but RF-DETR-small is 3; need to confirm which value `build_model(args_ckpt)`
  uses and whether `DETR_AUX_LOSS` in the YAML is ever passed to `args_ckpt`.

---

## Axis 2 — Feature Space

- Target files:
  - `models/rf_detr/backbone/dinov2.py:19-24` — `size_to_width["small"] = 384`
  - `models/rf_detr/backbone/dinov2_configs/dinov2_small.json:9` — `"hidden_size": 384`
  - `models/rf_detr/backbone/projector.py:141` — `MultiScaleProjector` class definition
  - `models/rf_detr/backbone/backbone.py:102` — projector instantiation, `out_channels=out_channels`
  - `models/rf_detr/lwdetr.py:321` — `out_channels=args.hidden_dim` (= 256 from config)
  - `models/rf_detr/lwdetr.py:172` — `out["outputs"] = hs[-1]`, shape=[B, num_queries, 256]
  - `configs/rf_motip_DT_V0motip.yaml:70` — `HIDDEN_DIM: 256`
  - `configs/rf_motip_DT_V0motip.yaml:52` — `FEATURE_DIM: 256`
  - `configs/rf_motip_DT_V0motip.yaml:54` — `ID_DIM: 256`
  - `models/motip/trajectory_modeling.py:21-25` — adapter FFN (detr_dim=256 in, 256 out)
  - `models/motip/id_decoder.py:59-65` — `embed_dim = feature_dim + id_dim = 512`
  - `models/motip/id_decoder.py:111-112` — tracklet concat: `cat([features, id_embeds], dim=-1)`
  - `train.py:730-756` — `prepare_for_motip()` assigns detector embeddings to trajectory slots

- Key question: Does the projector's `out_channels` equal exactly 256 at runtime, or does
  `args_ckpt.hidden_dim` differ from the YAML's `HIDDEN_DIM: 256`?

- Blocker: Confirmed at code level (YAML sets 256, projector uses `args.hidden_dim`).
  Runtime value requires checkpoint inspection to verify `args_ckpt.hidden_dim`.

---

## Axis 3 — Query Mechanism

- Target files:
  - `models/rf_detr/transformer.py:246-250` — top-K query selection by `max sigmoid(class_logit)`
  - `models/rf_detr/transformer.py:247` — `topk_proposals_gidx = torch.topk(enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1]`
  - `models/rf_detr/lwdetr.py:55-56` — `self.refpoint_embed`, `self.query_feat` (learnable query anchors)
  - `models/motip/__init__.py:42-47` — assertion: `Q == expected_Q` (300 queries enforced)
  - `train.py:712-756` — `prepare_for_motip()`: trajectory slot → detector output assignment
  - `train.py:733-735` — `detr_indices[flatten_idx]` used to extract matched embeddings
  - `train.py:740-741` — `trajectory_ann_idxs`, `unknown_ann_idxs` for per-frame GT correspondence
  - `train.py:751-752` — assigns matched embeddings to `trajectory_features`, `unknown_features`
  - `models/runtime_tracker.py:165-179` — inference: `output_embeds = detr_out["outputs"][0]`
  - `models/runtime_tracker.py:186-224` — inference slot assignment (no GT anchoring)
  - `models/motip/id_decoder.py:109` — `generate_empty_id_embed()` for newborn queries

- Key question: During training, frame-to-frame slot correspondence is anchored to GT
  annotations; at inference, what mechanism replaces this and how does query ordering
  interact with tracker state?

- Blocker: Full inference-time slot assignment logic requires tracing
  `runtime_tracker.py` thoroughly — the agent gave a partial view.

---

## Axis 4 — Matcher Alignment

- Target files:
  - `models/rf_detr/matcher.py:20-96` — `HungarianMatcher` class
  - `models/rf_detr/matcher.py:45` — `HungarianMatcher.forward()`
  - `models/rf_detr/lwdetr.py:243` — `indices = self.matcher(outputs_clean, targets)` inside DummyCriterion
  - `train.py:412` — `detr_criterion(outputs=detr_outputs, targets=..., batch_len=...)` returns `detr_indices`
  - `train.py:735` — `detr_indices[flatten_idx]` consumed in `prepare_for_motip()`
  - `models/motip/id_criterion.py:11-88` — `IDCriterion` class
  - `models/motip/id_criterion.py:25-53` — `IDCriterion.forward(id_logits, id_labels, id_masks)`
  - `models/motip/id_criterion.py:41-46` — loss computation (cross-entropy or focal)
  - `configs/rf_motip_DT_V0motip.yaml:84` — `DETR_SET_COST_CLASS: 0.0`  ← class cost is ZERO
  - `configs/rf_motip_DT_V0motip.yaml:85` — `DETR_SET_COST_BBOX: 5.0`
  - `configs/rf_motip_DT_V0motip.yaml:86` — `DETR_SET_COST_GIOU: 2.0`
  - `configs/rf_motip_DT_V0motip.yaml:80-82` — loss coefs: cls=2.0, bbox=5.0, giou=2.0
  - `configs/rf_motip_DT_V0motip.yaml:57` — `ID_LOSS_WEIGHT: 1`
  - `configs/rf_motip_DT_V0motip.yaml:59` — `USE_AUX_LOSS: False` (ID decoder aux loss)
  - `configs/rf_motip_DT_V0motip.yaml:61` — `USE_FOCAL_LOSS: False`

- Key question: With `DETR_SET_COST_CLASS: 0.0`, the Hungarian matcher matches detections
  to GT using only box L1 and GIoU costs — does this cause the matcher to produce
  systematically different assignments than MOTIP's original matcher (which uses classification cost)?

- Blocker: None at code level. The matcher cost configuration is confirmed in the YAML.

---

## Axis 5 — Script Inputs

- Embedding extraction call:
  - `models/rf_detr/lwdetr.py:172` — `out["outputs"] = hs[-1]` (shape [B, num_queries, 256])
  - Access pattern: `detr_out = model(frames=nested_tensor, part="detr")` then `detr_out["outputs"]`
  - Reference in inference code: `models/runtime_tracker.py:179` — `output_embeds = detr_out["outputs"][0]`

- Checkpoint loading pattern:
  - `models/misc.py:241-257` — `load_checkpoint(model, path, states=None, optimizer=None, scheduler=None)`
  - `torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)`
  - Checkpoint keys: `{"model": state_dict, "optimizer": ..., "scheduler": ..., "states": {...}}`
  - For MOTIP checkpoints saved by training: structure above
  - RF-DETR-only checkpoints: `{"model": state_dict, "args": argparse.Namespace(...)}`

- Sequence dir format (DanceTrack):
  - `{data_root}/DanceTrack/{split}/{sequence_name}/img1/{frame_idx+1:08d}.jpg`
  - Example: `DanceTrack/val/dancetrack0001/img1/00000001.jpg`
  - Annotation file: `{sequence_dir}/gt/gt.txt` (CSV: frame_id, obj_id, x, y, w, h, _, _, _)
  - Metadata: `{sequence_dir}/seqinfo.ini` → `[Sequence] imWidth, imHeight, seqLength`
  - Image loading in `data/dancetrack.py:61-67`

- Build pattern for script:
  ```python
  from models.motip import build as build_motip
  from models.misc import load_checkpoint
  model, _ = build_motip(config=config)
  load_checkpoint(model, path=checkpoint_path)
  model.eval()
  ```

- Blocker: The `build_motip(config)` call requires a full config dict matching YAML schema.
  For a standalone script, the user must either (a) pass a minimal config dict with at least
  `CKPT_PATH`, `HIDDEN_DIM`, `FEATURE_DIM`, `ID_DIM`, `NUM_ID_VOCABULARY`, `NUM_ID_DECODER_LAYERS`,
  `HEAD_DIM`, `REL_PE_LENGTH`, `FFN_DIM_RATIO`, `USE_AUX_LOSS`, `USE_SHARED_AUX_HEAD`,
  `USE_FOCAL_LOSS` or (b) load the config YAML directly.
  The exact required keys must be traced from `models/motip/__init__.py`.

---

## Open Blockers Summary

1. **Checkpoint `args_ckpt.dec_layers` value** — The YAML sets `DETR_DEC_LAYERS: 6` but
   RF-DETR-small is 3 layers. The DETR model is built from `args_ckpt` (from checkpoint file),
   not from YAML. Actual layer count requires reading the saved checkpoint. This affects
   Axis 1 (how many aux_outputs exist) and Axis 4 (how many matcher calls per forward).

2. **`args_ckpt.hidden_dim` vs YAML `HIDDEN_DIM`** — The YAML sets 256; checkpoint may differ.
   Affects Axis 2 projector output dimension at runtime.

3. **Joint training code path** — The current config has RF-DETR frozen (eval + no_grad).
   The joint training experiment (Epoch 4, HOTA=23.4) must use a different config or
   code path. This path is not visible in the current YAML and requires locating the
   joint training config to understand Axis 1 and 4 under gradient flow conditions.

4. **Inference-time slot assignment** — `runtime_tracker.py` slot assignment at inference
   (no GT anchoring) is partially explored. Full tracing needed for Axis 3 to confirm
   how query-to-tracklet mapping behaves without annotations.

5. **`build_motip()` required config keys** — For Axis 5 standalone script, the exact
   minimum config dict required by `models/motip/__init__.py` must be traced to avoid
   KeyError at script instantiation time.
```

---

## Implementation Steps (post-plan-mode)

1. Create `diagnostics/` directory
2. Write `diagnostics/investigation_plan.md` with content above
3. Commit to branch `claude/investigation-plan-gLMO8`
4. Push with `git push -u origin claude/investigation-plan-gLMO8`

## Verification

- `ls diagnostics/investigation_plan.md` — file exists
- `git log --oneline -1` — commit present on correct branch
- `git status` — clean tree
- Content review: all 5 axes present, each has file:line targets, key question, blocker

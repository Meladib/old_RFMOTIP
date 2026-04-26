# DIAG_MATCHER_ALIGNMENT

## 1. FINDINGS

- **Hungarian matcher runs exactly ONCE per forward pass.**
  `models/rf_detr/lwdetr.py:231-244`: `DummyCriterion.forward()` calls
  `self.matcher(outputs_clean, targets)` once on the full batch of flattened
  frames. There are no auxiliary-output matcher calls — the matcher is not
  applied to `aux_outputs` (which are never populated; see Axis 1).
  `train.py:412`: `detr_criterion(outputs=detr_outputs, targets=...) → detr_loss_dict, detr_indices`.

- **Matcher indices are the ONLY output of the detection stage; loss is zero.**
  `models/rf_detr/lwdetr.py:244`: `return {}, indices`.
  `DummyCriterion.weight_dict = {}`. The matcher runs purely to produce
  `detr_indices` for `prepare_for_motip`; no detection loss is computed.

- **Matcher operates on `pred_logits` and `pred_boxes` from the final decoder layer.**
  `models/rf_detr/lwdetr.py:238-243`:
  ```python
  outputs_clean = {
      "pred_logits": outputs["pred_logits"],
      "pred_boxes": outputs["pred_boxes"],
  }
  indices = self.matcher(outputs_clean, targets)
  ```
  `pred_logits` and `pred_boxes` are from `hs[-1]` (final layer), the same tensor
  fed to MOTIP as `f^n_t`. The matcher and MOTIP see the same feature layer —
  there is no layer mismatch between matching and feature extraction.

- **Matcher indices correctly flow into `prepare_for_motip`.**
  `train.py:412,422-426`:
  `detr_indices = detr_criterion(...)` → immediately used in `prepare_for_motip(
  detr_outputs=detr_outputs, annotations=annotations, detr_indices=detr_indices)`.
  No index reordering occurs between the matcher call and feature extraction.

- **`prepare_for_motip` correctly extracts GT-order embeddings.**
  `train.py:733-735`:
  ```python
  go_back_detr_idxs = torch.argsort(detr_indices[flatten_idx][1])
  detr_output_embeds = detr_outputs["outputs"][flatten_idx][
      detr_indices[flatten_idx][0][go_back_detr_idxs]]
  ```
  `detr_indices[flatten_idx][1]` = GT indices (sorted to GT order via `argsort`).
  `detr_indices[flatten_idx][0]` = matched prediction indices.
  Result: embeddings aligned to GT annotation order.
  The tensor fed to the ID decoder is `trajectory_features` and `unknown_features`
  built from these GT-matched embeddings — consistent with what the matcher saw.

- **Classification cost is ZERO in the YAML.**
  `configs/rf_motip_DT_V0motip.yaml:84`: `DETR_SET_COST_CLASS: 0.0`.
  However, `models/motip/__init__.py:141`: `build_motip_criterion(args_ckpt)`
  uses `args_ckpt` from the checkpoint, NOT the YAML. `args_ckpt.set_cost_class`
  is whatever RF-DETR was originally trained with.
  **The YAML `DETR_SET_COST_CLASS: 0.0` is never applied at runtime.**

- **`args_ckpt` matcher cost mismatch: YAML vs. checkpoint.**
  `models/motip/__init__.py:140-141`: only `args_ckpt.device` is patched from the
  runtime environment. All matcher cost parameters (`set_cost_class`,
  `set_cost_bbox`, `set_cost_giou`) come from the checkpoint's stored args.
  RF-DETR's original training would use focal-style class cost (typically 2.0).
  The YAML intention of `cost_class=0.0` (spatial-only matching) does not apply.

- **For single-class pedestrian tracking, classification cost matters less.**
  `models/rf_detr/matcher.py:73-81`: the class cost is a focal-style term
  `pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]`. With `num_classes=1`
  (after `reinitialize_detection_head(1)` at `motip/__init__.py:38`), all GT
  labels are class 0, and the matcher assigns the same class cost offset to all
  candidate predictions. The classification cost still influences the total cost
  matrix C but does not distinguish between GT objects of different categories —
  it only modulates the absolute scale of the cost.

- **Total detection loss magnitude vs. ID loss.**
  `train.py:484`: `loss = detr_loss + id_loss * id_criterion.weight`.
  `detr_loss = 0.0`; `id_criterion.weight = 1.0` (`ID_LOSS_WEIGHT: 1`).
  Effective loss = `id_loss × 1.0`. **Detection contributes zero gradient;
  all backprop flows exclusively through the ID decoder and trajectory modeling.**

- **ID criterion runs once, on the final ID decoder layer.**
  `configs/rf_motip_DT_V0motip.yaml:59`: `USE_AUX_LOSS: False`.
  `models/motip/id_decoder.py:183-186`: when `use_aux_loss=False`, returns
  `_unknown_id_logits` from the last iteration only (layer 2 of 3).
  `train.py:464`: `id_loss = id_criterion(id_logits, id_gts, id_masks)`.

- **Trajectory augmentation is active and correctly configured.**
  `configs/rf_motip_DT_V0motip.yaml:42-43`:
  `AUG_TRAJECTORY_OCCLUSION_PROB: 0.5`, `AUG_TRAJECTORY_SWITCH_PROB: 0.5`.
  Matches the paper's `λ_occ = λ_sw = 0.5`.

---

## 2. MISMATCH EVIDENCE

| Location | Observed | Expected | Impact |
|---|---|---|---|
| `lwdetr.py:231-244` | `DummyCriterion.weight_dict = {}` → detection loss = 0 | Detection loss to supervise box regression | Detector gets no gradient signal; detection quality cannot improve during training |
| `motip/__init__.py:140-141` | Only `args_ckpt.device` patched; `set_cost_class` from checkpoint | YAML `DETR_SET_COST_CLASS: 0.0` applied | Matcher uses original RF-DETR class cost (~2.0) not 0.0 — YAML intent silently ignored |
| `train.py:484` | `loss = 0 + id_loss × 1.0` | Balanced detection + ID loss | ID decoder receives all gradient; with frozen detector features, the only learned signal is ID vocabulary mapping |
| `yaml:59` `USE_AUX_LOSS: False` | ID decoder supervised at final layer only | Potentially aux-supervised at all 3 layers | Intermediate ID decoder layers get no direct supervision |

---

## 3. SEVERITY ASSESSMENT

**Critical**

The zero detection loss combined with the frozen detector means the entire training
signal is the ID criterion acting on frozen RF-DETR features — the system is
attempting to learn an identity-discriminative vocabulary mapping on top of
features optimized purely for detection, not for cross-frame identity. This
architectural gap directly explains the AssA collapse from the ~70 MOTIP baseline
to 20.5 (frozen) and 13.9 (joint training with a previous code version).

---

## 4. OPEN QUESTIONS

1. **What is `args_ckpt.set_cost_class`?**
   The actual value used by the matcher at runtime is from the checkpoint, not
   the YAML. Requires reading the checkpoint's stored args object.

2. **Does the matcher assignment quality degrade for crowded scenes?**
   With spatial-only matching (cost_class from checkpoint may not help), two
   overlapping pedestrians could produce ambiguous assignments, corrupting the
   GT-anchored feature extraction in `prepare_for_motip`.

3. **What loss weight would balance detection vs. ID loss in the joint training
   regime?**
   With 3 auxiliary detection losses active and λ_id=1.0, the effective
   detection-to-ID gradient ratio was approximately 3× (cls=2.0×3, bbox=5.0×3,
   giou=2.0×3 vs. id=1.0). Runtime analysis of gradient norms is needed.

---

## 5. HYPOTHESIS RANKING

#1 [Confidence: High] — The total training signal is ID loss only (detr_loss=0),
acting on frozen 256-dim features that are detection-optimized. The ID decoder
cannot learn reliable identity matching because the input features do not
discriminate identity across frames, causing AssA=20.5 — the ID decoder converges
but to a function that cannot generalize across frames.

#2 [Confidence: High] — The YAML `DETR_SET_COST_CLASS: 0.0` is silently ignored
because `build_motip_criterion(args_ckpt)` uses checkpoint args for all cost
parameters. The matcher is using non-zero class cost, which for single-class
tracking creates a uniform class cost offset that slightly perturbs the spatial
matching without improving assignment quality.

#3 [Confidence: Medium] — In the previous joint-training code version, the
effective loss was approximately `3 × detr_loss + 1 × id_loss` (due to 3 layers
with aux loss and no DummyCriterion), producing a gradient magnitude imbalance
that corrupted the detector faster than the ID decoder could learn, consistent
with the simultaneous drop in both DetA (72.9→40.1) and AssA (20.5→13.9).

# DIAG_DECODER_TOPOLOGY

## 1. FINDINGS

- **RF-DETR is hard-locked to eval + no_grad at every forward call.**
  `models/motip/motip.py:38-40` — `self.detr.eval()` followed immediately by
  `with torch.no_grad(): detr_out = self.detr(samples=frames)`.
  This is unconditional; no training config can override it.

- **`out["outputs"] = hs[-1]`** — only the final decoder layer output is consumed
  by MOTIP. `models/rf_detr/lwdetr.py:172`.
  With `dec_layers=3` (RF-DETR-small), `hs` has shape `[3, B, 300, 256]`;
  `hs[-1]` = layer-2 (zero-indexed) normalized output.

- **`return_intermediate=True` is set** in `models/rf_detr/transformer.py:157`
  (via `build_transformer(args)` at line 572).
  All three layer outputs are stacked into `hs`, but only `hs[-1]` is forwarded to
  MOTIP. Layers 0 and 1 are discarded.

- **aux_outputs are NEVER populated** in the integration.
  `models/rf_detr/lwdetr.py:154-155`: `if self.training and self.aux_loss:` —
  `self.training` is always `False` because the DETR is locked to eval.
  `DETR_AUX_LOSS: True` in the YAML has no runtime effect.

- **DummyCriterion returns zero detection loss.**
  `models/rf_detr/lwdetr.py:231-244`: `DummyCriterion.weight_dict = {}`
  (empty); `forward()` returns `{}, indices`.
  `train.py:473-481`: `detr_loss = sum(detr_loss_dict[k] * detr_weight_dict[k] ...)`
  iterates over an empty dict → `detr_loss = tensor(0.0)`.

- **`DETR_NUM_TRAIN_FRAMES: 0` (YAML)** means no frames are routed to the
  gradient-enabled DETR path. `train.py:331,388`:
  `detr_num_train_frames = min(0, _T)` → `if detr_num_train_frames > 0:` is
  never entered. All frames are processed under `torch.no_grad()`.

- **Two compounding locks:** (1) MOTIP.forward always wraps DETR in `no_grad`;
  (2) training loop never sends frames to a gradient path.
  Both are independently sufficient to freeze the detector.

- **Zero-decoder config does NOT silently reach MOTIP.**
  `models/rf_detr/transformer.py:267`: `if self.dec_layers > 0:` else `hs = None`.
  `models/rf_detr/lwdetr.py:172`: `out["outputs"] = hs[-1]` would raise
  `TypeError: 'NoneType' object is not subscriptable` if `dec_layers=0`.
  No silent degradation; the model would crash at build time.

- **YAML `DETR_DEC_LAYERS: 6` has no effect.**
  `models/motip/__init__.py:34`: `detr_model = build_model(args_ckpt)` uses
  `args_ckpt.dec_layers` from the saved RF-DETR checkpoint, not from the YAML.
  RF-DETR-small uses 3 decoder layers per the paper (Table 7).

- **The "joint training" experiment (Epoch 4, HOTA=23.4) is not reachable with
  the current code.** The `# 🔒 HARD LOCK RF-DETR TO EVAL` comment in motip.py
  suggests this lock was added after that experiment to prevent the observed
  gradient conflict from degrading detection.

---

## 2. MISMATCH EVIDENCE

| Location | Observed | Expected | Impact |
|---|---|---|---|
| `motip/motip.py:38-40` | `self.detr.eval()` + `no_grad` unconditionally | Gradient flow through detector for joint training | Joint training is architecturally impossible; only ID loss trains |
| `rf_detr/lwdetr.py:154-155` | `if self.training and self.aux_loss` — always False | `self.training` matches actual training mode | aux_outputs never generated; `DETR_AUX_LOSS: True` has zero effect |
| `rf_detr/lwdetr.py:231-244` | `DummyCriterion.weight_dict = {}` | Non-empty weight dict for detection loss | Detection loss is always zero; detector gets no gradient signal |
| `configs/rf_motip_DT_V0motip.yaml:67,75` | `DETR_AUX_LOSS: True`, `DETR_DEC_LAYERS: 6` | These values are used at runtime | Both are ignored; model built from checkpoint args with 3 layers |
| `rf_detr/lwdetr.py:172` | `hs[-1]` only | All layer outputs accessible | Intermediate layer embeddings inaccessible without hooking the decoder |

---

## 3. SEVERITY ASSESSMENT

**Critical**

The detector is completely frozen by two independent mechanisms (hard-lock in
motip.py and DummyCriterion zero-loss), so the HOTA drop from 70 → 23.4 cannot
be attributed to ID loss corrupting detection gradients as hypothesized — the
detector was never being trained. The observed degradation (frozen: 38.5,
joint: 23.4) must have occurred with a different code version where the lock was
absent.

---

## 4. OPEN QUESTIONS

1. **What code version produced the joint-training experiment?**
   The current motip.py has a "HARD LOCK" comment suggesting it was added after
   that experiment. Need to identify the commit that introduced the lock.

2. **What is `args_ckpt.dec_layers` in the saved checkpoint?**
   The YAML says 6 but RF-DETR-small is 3. Static analysis cannot confirm this
   without reading the checkpoint `args` object.

3. **Does `DETR_NUM_TRAIN_FRAMES > 0` in a different config enable gradient flow?**
   If so, does the motip.py hard-lock also need to be removed? The two locks are
   redundant but both would need changing.

---

## 5. HYPOTHESIS RANKING

#1 [Confidence: High] — The joint-training experiment used a code version without
the motip.py hard-lock, causing ID loss gradients to backprop through all 3
RF-DETR decoder layers simultaneously (3× detection gradient signal via
`return_intermediate=True`), overwhelming the single-point ID loss signal and
degrading both DetA and AssA.

#2 [Confidence: High] — The current frozen configuration never trains the
detector at all, so the AssA=20.5 ceiling in the frozen experiment reflects
purely the quality of frozen RF-DETR features for identity-discriminative tasks —
the ID decoder cannot overcome the feature quality limit.

#3 [Confidence: Medium] — Even in the hypothesized joint-training code, the
`DETR_AUX_LOSS: True` configuration with 3 decoder layers would apply detection
loss 3× per forward pass (aux_outputs[0], aux_outputs[1], pred_logits), while
ID loss applies once — the imbalance of 3:1 detection-to-ID gradient weighting
accelerates detector corruption.

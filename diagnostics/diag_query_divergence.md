# DIAG_QUERY_DIVERGENCE

## 1. FINDINGS

- **RF-DETR uses two-stage top-K query selection that changes per frame.**
  `models/rf_detr/transformer.py:246-247`:
  ```python
  topk_proposals_gidx = torch.topk(
      enc_outputs_class_unselected_gidx.max(-1)[0], topk, dim=1)[1]
  ```
  The top-300 proposals are selected from the encoder output by
  `max sigmoid(class_logit)` confidence score. This selection is computed fresh
  per forward pass; the slot indices of the 300 selected proposals will differ
  frame-to-frame as object positions and scores change.

- **During training, query-to-GT assignment is anchored via Hungarian matching.**
  `train.py:733-735`:
  ```python
  go_back_detr_idxs = torch.argsort(detr_indices[flatten_idx][1])
  detr_output_embeds = detr_outputs["outputs"][flatten_idx][
      detr_indices[flatten_idx][0][go_back_detr_idxs]]
  ```
  `detr_indices[flatten_idx]` is the matcher output `(pred_idx, gt_idx)`.
  For each GT object in each frame, the embedding is extracted by finding which
  query slot the Hungarian matcher paired with that GT. This is re-computed each
  frame — there is NO assumption that "slot N = object N" across frames.
  The per-frame re-matching means training is insulated from slot-ordering
  instability.

- **At inference, query ordering is bypassed via score filtering.**
  `models/runtime_tracker.py:165-179`: `_get_activate_detections` extracts all 300
  embeddings from `detr_out["outputs"][0]`, then filters by detection score
  threshold (`det_thresh=0.3`). The surviving set of embeddings is passed to
  `_get_id_pred_labels` which feeds the ID decoder.
  There is no slot-index-based lookup; identity assignment uses feature content
  via the ID decoder, not query slot position.

- **The slot ordering divergence problem is NOT directly a training-vs-inference
  mismatch.** Both training (`prepare_for_motip`) and inference
  (`_get_activate_detections`) bypass slot ordering. The deeper problem is that
  for the same physical object across two frames, different encoder slots may be
  selected as the "representative" query, producing embeddings from different
  learned slot positions.

- **NAS weight sharing makes slot semantics inconsistent.**
  From CLAUDE.md: RF-DETR was trained with architecture augmentation — at every
  iteration a random sub-network is sampled (1–6 decoder layers). All decoder
  layers share weights and must produce valid outputs for any depth configuration.
  No decoder layer position is specialized to represent objects at a fixed
  semantic depth. The features at layer-2 (final, consumed by MOTIP) have no
  stable positional semantics.

- **Query count assertion is structural only.**
  `models/motip/motip.py:42-47`:
  ```python
  Q = detr_out["pred_logits"].shape[1]
  assert Q == expected_Q
  ```
  This checks that 300 queries are returned but says nothing about ordering or
  cross-frame consistency.

- **Newborn handling is consistent regardless of query slot.**
  `models/motip/id_decoder.py:237-241`: `generate_empty_id_embed` uses
  `num_id_vocabulary * ones(...)` as the special token label for all unknown
  detections. This is independent of query slot index — every new detection gets
  the same `i_spec` embed, matching the MOTIP paper's newborn protocol.

- **Inference trajectory update uses ID-label matching, not slot matching.**
  `models/runtime_tracker.py:298-306`:
  ```python
  indices = torch.eq(current_id_labels[:, None], id_labels[None, :]).nonzero(...)
  current_features[current_idxs] = output_embeds[idxs]
  ```
  Features are stored in the trajectory buffer keyed by ID label, not by query
  slot index. A new frame's detection with ID `k` overwrites the stored feature
  for ID `k`. This is correct but relies on the ID decoder correctly assigning
  the same ID to the same object across frames, which in turn depends on the
  feature quality (see Axis 2).

---

## 2. MISMATCH EVIDENCE

| Location | Observed | Expected | Impact |
|---|---|---|---|
| `transformer.py:246-247` | Top-K selected by per-frame confidence scores | Stable slot ordering across frames | Different objects may be captured by different encoder slots per frame, producing inconsistent embeddings |
| `motip/motip.py:42-47` | Only count asserted (`Q == expected_Q`) | Cross-frame slot consistency verified | Slot reordering passes silently; no runtime signal of the instability |
| NAS weight sharing (paper) | All decoder layer depths share weights | Fixed-depth stable feature semantics | Layer-2 features have no specialized depth-position semantics; embeddings may vary with random NAS depth changes if any training is enabled |
| `runtime_tracker.py:294` | `current_features[current_idxs] = output_embeds[idxs]` | Stable per-slot feature retrieval | Feature update correctness depends entirely on ID decoder assignment quality |

---

## 3. SEVERITY ASSESSMENT

**High**

The per-frame top-K re-selection means the feature vector for the same physical
object can originate from different encoder memory slots across consecutive frames,
reducing the cross-frame cosine similarity of those feature vectors. This directly
reduces the ID decoder's ability to match `τ^n_t` (current) to `τ^m_k` (historical
tracklets), contributing to the AssA=20.5 in the frozen experiment and AssA=13.9
in joint training — a key structural driver of the HOTA drop from ~70 to 23.4.

---

## 4. OPEN QUESTIONS

1. **How large is the per-frame slot index variance for a static object?**
   If an object remains stationary, does the same encoder slot consistently win
   the top-K selection, or do nearby slots compete? This requires the Axis-5
   temporal stability script to measure empirically.

2. **Does the top-K encoder selection preserve spatial locality across frames?**
   For a slowly moving object, the encoder slot that wins the top-K competition
   should be spatially close across frames (similar deformable attention anchor
   points), which would limit embedding divergence even if the slot index changes.

3. **Do the learnable `refpoint_embed` and `query_feat` anchors (non-two-stage
   path) mitigate this for the non-two-stage RF-DETR variant?**
   If `args_ckpt.two_stage=False`, the slot ordering would be fixed (learnable
   positional anchors), but the paper uses two-stage for RF-DETR. Confirm from
   checkpoint.

---

## 5. HYPOTHESIS RANKING

#1 [Confidence: High] — The encoder top-K re-selection per frame causes the same
physical object to be represented by different encoder slots in consecutive frames,
producing embeddings with reduced cosine similarity and directly degrading the
ID decoder's cross-frame matching, explaining the AssA ceiling at 20.5.

#2 [Confidence: Medium] — NAS weight sharing means the 3-layer decoder produces
features that are not depth-position specialized: the features were trained under
random 1–6 layer configurations, so the final-layer embedding for the same object
differs between frames based on which random NAS depth was active during training,
increasing cross-frame feature variance beyond what is caused by top-K slot
instability alone.

#3 [Confidence: Low] — The training-time GT-anchored matching via `detr_indices`
(prepare_for_motip) teaches the ID decoder a feature space where GT-matched query
embeddings are used as `f^n_t`, while at inference the score-filtered embeddings
may include detection noise and miss GT objects — creating a domain gap between
training and inference feature distributions.

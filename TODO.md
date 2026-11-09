# TODO / Future Improvements

## GRPO reasoning-format learning

Done:
- Added a dedicated `reasoning` stage in `cloud_train.py` that trains the
  `think...` format on the `reasoning` subset of
  [`paperbd/paper_instructions_300K-v1`](https://huggingface.co/datasets/paperbd/paper_instructions_300K-v1/viewer/reasoning)
  before GRPO (`--reasoning-warmup-samples`, `--reasoning-warmup-epochs`).
- Ran the full pipeline on 2×T4. The format gap is fixed:
  warm-up 10,000 → 6,911 rows (1024-token filter), 3 epochs / 324 steps
  (~46 min); GRPO 1 epoch / 8 events (~50 min). Event-5 eval:
  `think_format_reward` **0.953**, datatype/schema match **93.8%**,
  `total_reward` **2.355**, best inference score **2.355**.

Remaining:
- Add a graded `think_format` reward in `reasoning/env.py` (partial credit for
  one opening/closing tag, then full credit for non-empty reasoning + response)
  to give GRPO a denser path to the format.
- Re-enable `--grpo-require-think-format` once the model reliably emits valid
  tags, so the semantic reward is gated as originally designed.
- Save a checkpoint at the end of GRPO training. `save_model()` currently runs
  only inside the eval block, so with 8 events and `eval_every=5` the final 3
  events are never written to disk (event-5 is the latest checkpoint).
- Delete the stale `models/reasoning_warmup_qwen06/checkpoint-3` from the HF
  repo; the new run wrote `checkpoint-250/300/324`.

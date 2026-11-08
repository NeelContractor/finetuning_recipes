# TODO / Future Improvements

## GRPO reasoning-format learning
- GRPO previously optimized semantic quality only: at the event-5 eval,
  `think_format_reward = 0.000` and datatype/schema match rates = `0.0%`.
  The model (Qwen3-0.6B) emitted no `<think>...</think>` blocks, so the format
  bonus never fired and there was no gradient toward the reasoning format.
- Root cause: the reasoning warm-up (`reasoning/sft_warmup.py`) ran only 3
  optimizer steps on 168 rows — far too little to teach the format.
- Done: `cloud_train.py` now has a dedicated `reasoning` stage that trains the
  format on the `reasoning` subset of
  `paperbd/paper_instructions_300K-v1`
  (https://huggingface.co/datasets/paperbd/paper_instructions_300K-v1/viewer/reasoning)
  before GRPO. Sizes are independent:
  - `--reasoning-warmup-samples` (default 10000, `-1` = all)
  - `--reasoning-warmup-epochs` (default 3)
  - `--grpo-samples` / `--grpo-eval-samples` for the GRPO rollouts/eval.
- Remaining:
  - Run `python cloud_train.py --stages reasoning grpo --num-gpus 2` without
    `--resume` (or with `--force`, which also clears the old 3-step warm-up
    output). Delete `models/grpo_qwen06` first if you want a clean GRPO run
    from the new warm-up. Confirm `think_format_reward` and datatype/schema
    match rates rise above 0%.
  - Add a graded `think_format` reward in `reasoning/env.py` (partial credit
    for exactly one opening/closing tag, then full credit for a non-empty
    reasoning + response), giving GRPO a dense path to the format.
  - Re-enable `--grpo-require-think-format` once the model reliably emits valid
    tags, so the semantic reward is gated as originally designed.

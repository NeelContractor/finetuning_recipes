# Reasoning training on Kaggle (2× T4)

Step-by-step runbook for training the reasoning-format warm-up (and optionally
GRPO) on Kaggle, starting from the SFT checkpoint already on the Hub.

> **Clone path matters.** Clone to `/kaggle/working/finetuning_recipes` exactly.
> The saved LoRA adapter configs reference that absolute path as their base
> model, so a different directory breaks the adapter merges.

## 0. Kaggle notebook setup

- **Accelerator:** Settings → Accelerator → **GPU T4 ×2**
- **Internet:** Settings → Internet → **On**
- **Token:** Add-ons → Secrets → create a secret named `HF_TOKEN` with a
  **write** token from <https://huggingface.co/settings/tokens>

Expose the token to all shell cells:

```python
# Cell 0
import os
from kaggle_secrets import UserSecretClient

os.environ["HF_TOKEN"] = UserSecretClient().get_secret("HF_TOKEN")
os.environ["HF_CHECKPOINT_REPO"] = "NeelContractor/finetuning-checkpoints"
print("token set:", bool(os.environ["HF_TOKEN"]))
```

## 1. Clone the GitHub repo

```bash
%%bash
rm -rf /kaggle/working/finetuning_recipes
git clone https://github.com/NeelContractor/finetuning_recipes.git /kaggle/working/finetuning_recipes
cd /kaggle/working/finetuning_recipes
git log --oneline -1
nvidia-smi -L
```

`nvidia-smi -L` should list **two** Tesla T4 GPUs.

## 2. Pull the SFT checkpoint from HF

Only `models/sft_qwen06_merged` is required: the reasoning warm-up starts from
the SFT merged model.

```python
# Cell 2a — download the SFT merged base (~1.2 GB)
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NeelContractor/finetuning-checkpoints",
    repo_type="model",
    local_dir="/kaggle/working/finetuning_recipes",
    allow_patterns=["models/sft_qwen06_merged/*"],
)
```

```bash
%%bash
# Cell 2b — verify
ls -lh /kaggle/working/finetuning_recipes/models/sft_qwen06_merged
```

You should see `model.safetensors`, `config.json`, and the tokenizer files.

> **Do not** download `models/reasoning_warmup_qwen06/*` for a fresh run. It
> contains the old 3-step checkpoint, which `--resume` would pick up.

## 3. Optional: smoke test (10–15 min)

```bash
%%bash
cd /kaggle/working/finetuning_recipes
python cloud_train.py --stages reasoning grpo --num-gpus 2 --smoke --install
```

Verified on 2×T4: 2000 warm-up rows → 582 filtered (>1024 tok) → 1389 kept,
22 steps, 177 s (~8 s/it including startup); GRPO ran 4 events. Rewards were
non-degenerate but `think_format_reward` stayed `0.000` — expected at this
scale; the full 3-epoch warm-up is the real test.

If you run the smoke test, clean its outputs before the real run:

```bash
%%bash
cd /kaggle/working/finetuning_recipes
rm -rf \
  models/reasoning_warmup_qwen06 \
  models/reasoning_warmup_qwen06_merged \
  models/grpo_qwen06 \
  models/grpo_qwen06_merged \
  data/reasoning \
  data/reasoning_warmup
```

## 4. Train reasoning (and GRPO)

Run both stages in one command so the warm-up is trained once and reused:

```bash
%%bash
cd /kaggle/working/finetuning_recipes
python cloud_train.py --stages reasoning grpo --num-gpus 2 --install
```

- Warm-up only (~35–50 min): replace the stage list with `--stages reasoning`.
- Expected outputs: `models/reasoning_warmup_qwen06/final`,
  `models/reasoning_warmup_qwen06_merged`, then `models/grpo_qwen06/llm*`.
  GRPO is **not** auto-merged — see step 5.
- Drop `--install` on later runs in the same session (the smoke test already
  installed everything).
- **Do not** use `--stages grpo` alone; it re-runs the warm-up by design.
- If you ran the smoke test in this session, either clean its output (step 3)
  or add `--force`. Without it, `prepare_reasoning_data` reuses the 2000-row
  warm-up and 32/8-row GRPO slices, and the "full" run stays tiny.

## 5. Verify the outputs

```bash
%%bash
cd /kaggle/working/finetuning_recipes
ls -lh models/reasoning_warmup_qwen06 models/reasoning_warmup_qwen06_merged
ls -lh models/grpo_qwen06
cat models/grpo_qwen06/logs/train_metrics.txt
```

Look for `rewards/think_format_reward/mean` > 0 in the metrics. If it stays
0.000 after the full warm-up, inspect
`models/grpo_qwen06/logs/eval_generations_*.json` before spending time on GRPO.

Cloud_train does not merge the GRPO adapter. To produce the merged GRPO model
(used by the upload in step 6), run:

```bash
%%bash
cd /kaggle/working/finetuning_recipes
python instruction_tuning/merge_adapter.py \
  --adapter_path models/grpo_qwen06/llm_best_0 \
  --output_path models/grpo_qwen06_merged
```

Optional format check on the warm-up adapter:

```bash
%%bash
cd /kaggle/working/finetuning_recipes
python reasoning/eval_checkpoint.py \
  --adapter_path models/reasoning_warmup_qwen06/final \
  --base_model models/sft_qwen06_merged \
  --dataset_name data/reasoning \
  --num_eval_rows 16 \
  --max_new_tokens 256
```

## 6. Push checkpoints to HF

This matches the existing repo layout (`models/...`) and includes the merged
models:

```python
# Cell 6 — upload adapters, checkpoints, merged models, and logs
from huggingface_hub import HfApi

api = HfApi()
repo = "NeelContractor/finetuning-checkpoints"
root = "/kaggle/working/finetuning_recipes"

folders = [
    "models/reasoning_warmup_qwen06",
    "models/reasoning_warmup_qwen06_merged",
    # include these if you ran GRPO:
    "models/grpo_qwen06",
    "models/grpo_qwen06_merged",
]

for folder in folders:
    path = f"{root}/{folder}"
    try:
        api.upload_folder(
            folder_path=path,
            repo_id=repo,
            repo_type="model",
            path_in_repo=folder,
            commit_message=f"Upload {folder}",
        )
        print("uploaded:", folder)
    except FileNotFoundError:
        print("skipped (missing):", folder)
```

If this raises `XetAuthorizationError` or the `huggingface_hub.errors`
`ImportError`, pin the Hub client and rerun the upload cell in a fresh process:

```bash
%%bash
pip install -q --force-reinstall --no-deps huggingface_hub==0.36.2
```

### Optional: crash-safe uploads during training

Add these flags to the step 4 command:

```bash
--push-to-hub --hub-repo NeelContractor/finetuning-checkpoints \
--no-hub-private --hub-upload-every-saves 5
```

> Caveat: the built-in pusher writes to `<repo>/reasoning_warmup_qwen06/...`
> (no `models/` prefix) and does **not** upload the merged dir. Still run step 6
> afterward for the final layout.

## 7. Resume after a Kaggle session restart

Re-run cells 0–2, then download the partial run and resume:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NeelContractor/finetuning-checkpoints",
    repo_type="model",
    local_dir="/kaggle/working/finetuning_recipes",
    allow_patterns=[
        "models/sft_qwen06_merged/*",
        "models/reasoning_warmup_qwen06/*",
    ],
)
```

```bash
%%bash
cd /kaggle/working/finetuning_recipes
# If the old 3-step checkpoint came back, remove it so Trainer does not resume from it:
rm -rf models/reasoning_warmup_qwen06/checkpoint-3
python cloud_train.py --stages reasoning --num-gpus 2 --resume
```

> Do **not** add `--force` when resuming; it deletes the warm-up output.

## Timing (2× T4)

| Phase | Estimate |
|---|---|
| Install (`--install`) | 5–10 min |
| Reasoning warm-up (`--stages reasoning`) | 35–50 min |
| GRPO (`--stages grpo`) | 1.5–2 h |
| Merges + data downloads | 5–10 min |
| Full `--stages reasoning grpo` | ~2.5–3 h |

The warm-up estimate uses the smoke measurement: 1389 rows → 22 steps → 177 s
(~8 s/it incl. startup). Full scale is ~7k rows after the 1024-token filter,
3 epochs ≈ 330 steps.

## Troubleshooting

- **`--num-gpus 2` fails:** the runtime is not `GPU T4 ×2`. Change it in Settings.
- **Dataset/reward-model downloads fail:** turn Internet on.
- **Adapter merge looks at a wrong path:** you cloned somewhere other than
  `/kaggle/working/finetuning_recipes`.
- **Upload fails:** pin `huggingface_hub==0.36.2` (see step 6).
- **GRPO unexpectedly retrains the warm-up:** `--stages grpo` always runs the
  warm-up first; use `--stages reasoning grpo`.
- **Resume starts from step 3:** delete
  `models/reasoning_warmup_qwen06/checkpoint-3` (step 7).

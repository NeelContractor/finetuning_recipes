# Running the Full Fine-Tuning Pipeline on Google Colab (Free T4)

> **Automated runner:** `cloud_train.py` wraps these steps (data prep → CPT →
> SFT → DPO → GRPO) into one script. Recommended first run from the repo root:
>
> ```bash
> python cloud_train.py --install --smoke
> ```
>
> On Kaggle with **GPU T4 x2**, add `--num-gpus 2` to train with both GPUs via DDP:
>
> ```bash
> python cloud_train.py --install --num-gpus 2 --smoke
> ```
>
> Run `python cloud_train.py --help` for all options. The manual steps below are
> still useful for debugging individual stages.

This guide runs the **entire** pipeline — data prep → CPT → SFT → DPO/ORPO → reward model → GRPO — on a **free Google Colab T4 GPU (16 GB VRAM)**.

Unlike your 8 GB M2 (where unsloth is CUDA-only and the SFT script won't run), the **T4 is CUDA**, so:

- ✅ All 5 training scripts take the **Unsloth CUDA path** automatically (`torch.cuda.is_available()` → `True`)
- ✅ The unconditional `from unsloth import ...` at the top of each script works fine
- ✅ `load_in_4bit=True` and `optim="adamw_8bit"` actually activate
- ✅ Flash Attention and fp16 mixed precision work

Your tiny 0.6B model (`Qwen/Qwen3-0.6B`) is trivially small for 16 GB VRAM. The same scripts run **verbatim** from the repo — the only new work is **Colab-specific setup** (install unsloth, mount Drive, adjust for the ephemeral VM, size the datasets to fit the session).

---

## 0. The 3 Colab gotchas (read this first)

1. **The VM is disposable.** Every file on the VM disk (`/content/...`) is **deleted when the session ends** (cap ~12 h, idle timeout ~90 min). You **must**:
   - Mount Google Drive and do all project work under `/content/drive/MyDrive/...`
   - Copy each stage's `models/.../final` checkpoint to Drive as soon as training finishes
2. **unsloth is not preinstalled.** Install it in a cell at the start of the runtime (one time).
3. **Free tier limits:** no terminal/background execution (Pro+ only), GPU not guaranteed at peak times (may get a K80 12GB — still fine for 0.6B). Request T4 during off-peak hours (late night US).

---

## 1. Create the notebook & select the T4

1. Go to https://colab.research.google.com
2. **File → New notebook**
3. **Runtime → Change runtime type → Hardware accelerator: T4 GPU**
4. Verify the GPU in a cell:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

You should see `Tesla T4` with ~15 GB usable VRAM.

---

## 2. Mount Drive & install dependencies

### 2a. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2b. Install unsloth + missing packages

Run this once per runtime. Install in this order (unsloth first, then the rest):

```python
!pip install unsloth

!pip install datasets==3.6 arxiv pypdf rich outlines rouge-score sacrebleu \
    sentence-transformers umap-learn plotly scipy scikit-learn \
    peft trl bert-score gradio
```

> **Note on `torch`/`transformers`:** Colab already ships `torch` and `transformers`. To avoid breaking the Colab runtime, **do not** reinstall them unless unsloth forces a specific version. Unsloth usually installs its own compatible `transformers` — that's fine.

---

## 3. Get the project code onto Drive

### 3a. Pull the repo (or upload your edited copy)

If you have a git repo (recommended so you can track edits):

```python
%cd /content/drive/MyDrive
!git clone https://github.com/<you>/fine_tuning.git
%cd fine_tuning
```

> If you made local edits (like the README / model IDs), the cleanest path is to **commit + push those first**, then clone here. Or upload your whole `fine_tuning/` folder to Drive via the Files panel and use that as the working dir.

### 3b. (Optional) Create the `.env` with your OpenRouter key

Only needed for the **LLM-judge** scripts. The free judge model is already the default in the code, and the key can also be set directly in Python below. Simplest in Colab:

```python
import os
os.environ["OPENROUTER_API_KEY"] = "sk-or-..."   # paste your key (never commit it)
```

---

## 4. Working directory convention

All stages run from the project root **on Drive** so nothing is lost:

```python
import os
os.chdir("/content/drive/MyDrive/fine_tuning")
print(os.getcwd())
```

Every `models/...` output therefore lands in `/content/drive/MyDrive/fine_tuning/models/...` — persistent and safe.

> **Pro tip:** run `%cd` / `os.chdir` after every VM restart, and **re-run the Step 2 install cells** on each new session. Consider putting cells 2a–4 into a single "setup" cell block you can re-run in one click.

---

## 5. Full pipeline (step by step, copy-paste per cell)

Each numbered block below is a single Colab cell. Model = `Qwen/Qwen3-0.6B`. Runs are sized to fit comfortably inside a single free session.

### Step 5.1 — Prepare CPT data (arXiv papers)

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python data_prep/cpt/download_arxiv_files.py
```

Download a handful of papers (keep it small for the 12 h budget — a few dozen papers is plenty for a demo).

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python data_prep/cpt/get_dataset.py data_prep/cpt/papers
```

This creates `cpt_train_dataset_N.jsonl` and `cpt_val_dataset_50.jsonl` in the project root.

> Pick a **small, real-world dataset**, e.g. your own domain papers, or keep the arxiv download small. The full pipeline already has multiple stages; don't blow the whole session on data prep.

---

### Step 5.2 — Continued Pre-Training (CPT)

On the T4, `cpt/sft.py` auto-detects CUDA and uses **Unsloth + 4-bit quantization**. Slightly larger batch is fine on 16 GB.

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python cpt/sft.py \
  --base_model_id Qwen/Qwen3-0.6B \
  --output_model_id cpt_arxiv_qwen06 \
  --dataset_path cpt_train_dataset_N.jsonl \
  --test_dataset_path cpt_val_dataset_50.jsonl \
  --max_seq_length 512 \
  --batch_size 8 \
  --epochs 10
```

Replace `cpt_train_dataset_N.jsonl` with the exact filename printed in 5.1.

**Save progress to Drive immediately after training** (it's already on Drive under `models/`, but double-check):

```python
%%bash
ls -la /content/drive/MyDrive/fine_tuning/models/cpt_arxiv_qwen06/final
```

**Verify / evaluate:**

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python cpt/inference.py \
  --models "models/cpt_arxiv_qwen06/final" "Qwen/Qwen3-0.6B" \
  --base_model Qwen/Qwen3-0.6B \
  --dataset cpt_val_dataset_50.jsonl \
  --num_samples 10
python cpt/evals.py --input_json inference_generations.json
```

---

### Step 5.3 — Instruction Tuning (SFT)

**This step is the one that breaks on your Mac** — but it runs fine on the T4 because it uses unsloth directly. (`instruction_tuning/sft.py` has no Mac fallback, but it doesn't need one here.)

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python instruction_tuning/sft.py \
  --base_model_id models/cpt_arxiv_qwen06/final \
  --output_model_id instruction_tuned_qwen06 \
  --dataset paperbd/paper_instructions_300K-v1 \
  --max_seq_length 1024 \
  --batch_size 4 \
  --grad_accum 8 \
  --epochs 3
```

> **Dataset size warning:** `paper_instructions_300K-v1` is 300K rows — too big for one free session (and way more than a 0.6B learning model needs). **Sub-sample it** in the cell first, then pass your own dataset:

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("paperbd/paper_instructions_300K-v1", split="train")
ds = ds.select(range(20000))          # 20K rows is plenty for 0.6B
ds.to_json("paper_instructions_20k.jsonl")
print("wrote paper_instructions_20k.jsonl", len(ds))
EOF
```

Then run SFT with `--dataset paper_instructions_20k.jsonl` (a local JSONL path works with the loader). If still too slow, trim to 5–10K.

**Merge the LoRA adapter** (needed for inference/export):

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python instruction_tuning/merge_adapter.py \
  --adapter_path models/instruction_tuned_qwen06/final \
  --output_path models/instruction_tuned_qwen06_merged
```

**Chat / inference (transformers path — no MLX on T4):**

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python instruction_tuning/single_inference.py \
  -m models/instruction_tuned_qwen06_merged \
  -p "What is self-attention?"
```

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python instruction_tuning/inference.py -m models/instruction_tuned_qwen06_merged -n 50
```

**LLM-judge the evals (needs OpenRouter key; free judge model is the default):**

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python instruction_tuning/api_inference.py -m "nvidia/nemotron-3-super-120b-a12b:free" -n 50
python instruction_tuning/llm_judge.py -i instruction_tuning/evals/<your_file>_api_results.jsonl
python instruction_tuning/print_judge_scores.py
```

---

### Step 5.4 — Preference Optimization (DPO / ORPO)

**5.4a. Generate multiple responses per prompt (temperature sweep).** On the T4 use the **transformers path (no `--mlx`)**:

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python preference_optimization/diversity_evals/generate_responses.py \
  -m models/instruction_tuned_qwen06_merged \
  --num_samples 100 --n_responses 4 --temperatures 0.3 0.5 0.7 1.0
```

Output: a `*_4r_temp*.jsonl` file (e.g. `pref_dataset.jsonl`).

**5.4b. Rank the responses** (either OpenRouter with the free judge, or the transformers/outlines path):

```python
# Option A — OpenRouter API (free judge model, needs OPENROUTER_API_KEY)
%%bash
cd /content/drive/MyDrive/fine_tuning
python preference_optimization/generate_ranked_pairs_openai.py -i pref_dataset.jsonl

# Option B — local (outlines structured gen, transformers on T4)
cd /content/drive/MyDrive/fine_tuning
python preference_optimization/generate_ranked_pairs.py -i pref_dataset.jsonl
```

Both write `<input>_ranked.jsonl`.

**5.4c. Convert to pairwise DPO data:**

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python preference_optimization/build_dpo_dataset.py -i pref_dataset_ranked.jsonl
```

**5.4d. Train DPO or ORPO** on the SFT model (16 GB → can raise from the 8 GB flags):

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python preference_optimization/train_preference.py \
  --method dpo \
  --base_model_id models/instruction_tuned_qwen06/final \
  --output_model_id dpo_qwen06 \
  --dataset pref_dataset_ranked_dpo.jsonl \
  --max_seq_length 1024 --batch_size 4 --grad_accum 8 --epochs 3
```

---

### Step 5.5 — Reward Model (optional)

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python reward_models/train_reward_model.py --output qwen06_v1
# → models/reward_model_finetuned_qwen06_v1
```

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python preference_optimization/evaluate_reward_model.py \
  --model models/reward_model_finetuned_qwen06_v1
```

Note: the reward model uses a MiniLM encoder — small and fast on the T4. On free tier, `torch.backends.mps` is absent, so it falls back to `cuda`.

---

### Step 5.6 — Reasoning / RLVR (GRPO)

**5.6a. SFT warm-up** for the reasoning format:

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python reasoning/sft_warmup.py \
  --base_model_id models/instruction_tuned_qwen06/final \
  --dataset paperbd/paper_instructions_300K-v1 \
  --dataset_config reasoning \
  --output_model_id reasoning_warmup_qwen06 \
  --max_seq_length 1024 --batch_size 4 --grad_accum 8 --epochs 3
```

> Sub-sample the reasoning dataset too (see Step 5.3) — pass a small local JSONL and lower `--epochs`.

**5.6b. Edit the GRPO config** (`reasoning/grpo/config.yaml`) to point at your model and fit the session:

```yaml
model:
  name: "models/merged/instruction_tuned_qwen06"   # merged SFT base
  from_sft: false
  max_new_tokens: 512

training:
  rollout_batch_size: 8
  n_rollouts: 4
  batch_size: 8
  gradient_accumulation_steps: 4
  buffer_size: 256

data:
  dataset: "paper_instructions_20k"   # your subsampled path or HF name
  train_data_size: 5000
  eval_data_size: 256
```

**5.6c. Run GRPO:**

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
python reasoning/grpo/train.py reasoning/grpo/config.yaml -o reasoning_qwen06
```

> GRPO is the heaviest stage and the most likely to exceed the free session. Keep the rollout size small and lower `train_data_size` / `num_epochs` in the config if it's too slow. If it won't fit in one session, split it and rely on saved checkpoints.

---

## 6. Persist between sessions (IMPORTANT)

Because the VM is wiped on disconnect, before closing any session run this to confirm your work is safe on Drive:

```python
%%bash
cd /content/drive/MyDrive/fine_tuning
du -sh models/ data_prep/*.jsonl cpt_*.jsonl paper_*.jsonl 2>/dev/null
ls -R models | head -60
```

Everything under `/content/drive/MyDrive/fine_tuning/` persists. **On your next session**, just re-run:
1. `drive.mount(...)`
2. the `!pip install unsloth ...` cell
3. `os.chdir("/content/drive/MyDrive/fine_tuning")`

then continue from the next stage.

---

## 7. Troubleshooting

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Raise `grad_accum`, lower `--batch_size` to 2; ensure 4-bit is active (CUDA path). 0.6B rarely hits this. |
| GPU is K80, not T4 (12 GB) | Reconnect Runtime → "Change runtime type" → T4 (do this off-peak / late night US). Still fine for 0.6B. |
| Session disconnected mid-training | Re-run setup cells, continue from last Drive checkpoint. Accept the 12 h cap. |
| `unsloth` import/version error | `!pip uninstall -y unsloth transformers trl` then `!pip install unsloth` to rebuild a compatible set. |
| `datasets==3.6` conflicts with unsloth | Let unsloth's `datasets` version win; re-run the other `pip install` lines only if a package errors. |
| LLM-judge 429 (rate limit) | Free model is rate-limited (~20 req/min). Add `--limit` / smaller `-n`, or wait between runs. |
| Out of free GPU quota for the week | Colab caps free GPU hours (~15–30 h/wk, dynamic). Spread stages across days. |
| Training too slow / won't finish | Sub-sample datasets harder, lower `--epochs`, use 0.6B only. The pipeline has many stages — keep each run small. |

---

## 8. The whole pipeline at a glance

```
Data prep → CPT → (merge) → SFT → (merge) → DPO/ORPO → Reward → GRPO
 5.1        5.2        5.3      5.3            5.4        5.5     5.6
```

**Golden rules on free T4:**
- Everything on Drive (VM is ephemeral)
- Install unsloth once per runtime, re-run after each restart
- Keep data small (sub-sample) so each stage fits in a ~12 h session
- Save/verify checkpoints on Drive after every stage
- Off-peak hours → more reliable T4 allocation

This pipeline now runs end-to-end on the free T4 without any of the Mac/MLX workarounds.

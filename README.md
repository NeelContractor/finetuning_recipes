# Fine-Tuning Recipes

End-to-end post-training recipes for small language models — continued pre-training (CPT), supervised fine-tuning (SFT), preference alignment (DPO/ORPO, reward models), and reasoning (format warm-up + GRPO).

This project is based on the [finetuning_recipes](https://github.com/avbiswas/finetuning_recipes) repo from the Neural Breakdown YouTube course. It runs on **macOS (Apple Silicon via MLX)**, Linux (CUDA via Unsloth), and Windows/CPU.

> **This README is written specifically for an 8 GB MacBook Air (M2).** It uses a tiny model (`Qwen/Qwen3-0.6B`, ~600M params) and settings tuned so every stage fits comfortably in 8 GB of unified memory. If you have more RAM or an NVIDIA GPU, you can raise the sizes.

---

## 1. What each stage does

| Stage | Directory | What it does |
|-------|-----------|--------------|
| **Data prep** | `data_prep/cpt/` | Download + clean arXiv papers into train/val JSONL |
| **CPT** (Continued Pre-Training) | `cpt/` | Keep pre-training the base model on domain data (papers) |
| **SFT** (Supervised Fine-Tuning) | `instruction_tuning/` | Teach the model to answer instructions (ChatML format) |
| **DPO / ORPO** | `preference_optimization/` | Align with human preference pairs / ranked responses |
| **Reward model** | `reward_models/` | Train a scorer that rates "how good" an answer is |
| **Reasoning format** | `reasoning/sft_warmup.py` | Teach the `<think> ... </think>` format before RL |
| **GRPO / Reasoning** | `reasoning/grpo/` | RL with verifiable rewards (reward model + format/schema checks) |
| **Diversity evals** | `preference_optimization/diversity_evals/` | Measure response diversity (EAD, SBERT, Vendi) |

**The whole pipeline in one line:** `pre-train on papers → follow instructions → learn preferences → score answers → learn the reasoning format → reason (GRPO)`.

> **Running on a GPU (Kaggle/Colab)?** [`cloud_train.py`](cloud_train.py) is a
> one-command orchestrator for the full pipeline — data prep → CPT → SFT → DPO →
> reasoning-format warm-up → GRPO — with DDP, checkpoint resume, smoke tests,
> and Hub uploads. See [COLAB.md](COLAB.md).
>
> ```bash
> python cloud_train.py --install --num-gpus 2
> # retrain the reasoning format, then GRPO:
> python cloud_train.py --stages reasoning grpo --num-gpus 2
> ```

---

## 2. Prerequisites

- A Mac with **Apple Silicon** (M1/M2/M3...) and **at least 8 GB RAM**
- Python 3.12 (pinned by `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) — fast Python package manager
- (Optional) A free [OpenRouter](https://openrouter.ai/) API key, used only by the **LLM-judge** eval scripts

---

## 3. One-time setup

```bash
# 1. Enter the project
cd fine_tuning

# 2. Install all dependencies into a virtualenv (torch, transformers, trl, mlx, mlx-lm, outlines, datasets, ...)
uv sync

# 3. Create your local env file and (optionally) paste your OpenRouter key
cp .env.example .env

# 4. Activate the environment
source .venv/bin/activate
```

**How it runs on your Mac:**
- **Training** (`cpt/sft.py`, `instruction_tuning/sft.py`, `preference_optimization/train_preference.py`) uses **Unsloth**, which is **CUDA-only** — on Apple Silicon these scripts automatically fall back to standard **Hugging Face Transformers + PEFT (LoRA)**. Everything goes through your M2's GPU.
- **Fast local inference** (ranking, chat, evals) uses **MLX** (`mlx_lm`), the native Apple Silicon runtime.

> Verify your environment with `./check_env.sh` (it mainly targets CUDA boxes, but confirms `python`/`uv` are on PATH).

---

## 4. The model we use

For the whole project we use the tiny **0.6B** Qwen model:

| Purpose | Model ID | Notes |
|---------|----------|-------|
| Base for **all training** (CPT → SFT → DPO / reasoning → GRPO) | `Qwen/Qwen3-0.6B` | HF Transformers/PEFT base (~600M params) |
| Local **MLX inference** (ranking, chat, evals) | `mlx-community/Qwen3-0.6B-4bit` | 4-bit MLX quant (~350 MB) — fits 8 GB easily |
| OpenRouter **LLM judge** (free) | `nvidia/nemotron-3-super-120b-a12b:free` | Default judge model in the code |

> **RAM budget (8 GB):** the 0.6B model in full precision ≈ 1.2 GB; LoRA training on top adds a few hundred MB. This leaves room for macOS + a terminal. Do **not** use the 4B/8B models here — they'll swap memory and slow your M2 to a crawl.

---

## 5. Full step-by-step walkthrough

### Step 1 — Prepare CPT data (arXiv papers)

Download a few arXiv papers as `.txt`, then turn them into `cpt_train_dataset.jsonl` / `cpt_val_dataset_50.jsonl`.

```bash
# 1a. Download papers (writes .txt files under data_prep/cpt/papers/)
uv run python data_prep/cpt/download_arxiv_files.py

# 1b. Split into train + 50-sample val JSONL (removes references/appendix sections)
uv run python data_prep/cpt/get_dataset.py data_prep/cpt/papers
```

This produces:
- `cpt_train_dataset_N.jsonl` (all papers except 50)
- `cpt_val_dataset_50.jsonl` (50 held-out papers)

> Optional alternative cleaners: `data_prep/cpt/build_text_jsonl.py <in_dir> <out.jsonl>` and `.../build_scientific_archive_text_jsonl.py <archive.jsonl> <out.jsonl>`. For a quick start the `download_arxiv_files.py` + `get_dataset.py` pair is enough.

---

### Step 2 — Continued Pre-Training (CPT)

Now we keep pre-training the tiny base model on the papers. **8 GB-friendly settings** (small seq len + small LoRA) are spelled out in the flags:

```bash
# LoRA continued pre-training on Qwen3-0.6B
uv run python cpt/sft.py \
  --base_model_id Qwen/Qwen3-0.6B \
  --output_model_id cpt_arxiv_qwen06 \
  --dataset_path cpt_train_dataset_N.jsonl \
  --test_dataset_path cpt_val_dataset_50.jsonl \
  --max_seq_length 384 \
  --batch_size 8 \
  --epochs 10
```

> Replace `cpt_train_dataset_N.jsonl` with the exact filename that `get_dataset.py` printed (e.g. `cpt_train_dataset_212.jsonl`).

Check what it learned by generating text and computing ROUGE/BLEU/BERTScore:

```bash
# Batch generation + perplexity across checkpoints (compare CPT vs base)
uv run python cpt/inference.py \
  --models "models/cpt_arxiv_qwen06/final" "Qwen/Qwen3-0.6B" \
  --base_model Qwen/Qwen3-0.6B \
  --dataset cpt_val_dataset_50.jsonl \
  --num_samples 10

# Metrics only, from the saved generations file
uv run python cpt/evals.py --input_json inference_generations.json
```

**Where it lands:** `models/cpt_arxiv_qwen06/final`

> The CPT checkpoint is a good *base for SFT*. For the course you can also just download a pre-trained instruction model — but training it yourself is the whole point.

---

### Step 3 — Instruction Tuning (SFT)

Teach the CPT base to answer instructions in ChatML format using an instruction dataset.

```bash
# 3a. SFT the CPT model on the paper-instructions dataset (8 GB-safe flags)
uv run python instruction_tuning/sft.py \
  --base_model_id models/cpt_arxiv_qwen06/final \
  --output_model_id instruction_tuned_qwen06 \
  --dataset paperbd/paper_instructions_300K-v1 \
  --max_seq_length 1024 \
  --batch_size 4 \
  --grad_accum 8 \
  --epochs 3 \
  --learning_rate 2e-4
```

> `batch_size 4 × grad_accum 8` keeps the effective batch at 32 while using little memory — the key trick for training on 8 GB. `max_seq_length 1024` (instead of 2048) halves the memory again.

**3b. Merge the LoRA adapter into a full model** (required before MLX export / fast inference):

```bash
uv run python instruction_tuning/merge_adapter.py \
  --adapter_path models/instruction_tuned_qwen06/final \
  --output_path models/instruction_tuned_qwen06_merged
```

**3c. Export to MLX** so inference runs fast on your M2:

```bash
uv run python mlx_export.py --adapter_path models/instruction_tuned_qwen06/final --q_bits 4
# → merged:  models/merged/instruction_tuned_qwen06
# → MLX:     models/mlx/instruction_tuned_qwen06
```

Then evaluate / chat:

```bash
# Single-turn test
uv run python instruction_tuning/single_inference.py \
  -m models/instruction_tuned_qwen06_merged \
  -p "What is self-attention?"

# Interactive chat using the exact training prompts (MLX, fast on M2)
uv run python instruction_tuning/task_chat.py -m models/mlx/instruction_tuned_qwen06

# Batch eval (transformers or MLX)
uv run python instruction_tuning/inference.py -m models/instruction_tuned_qwen06_merged -n 50
uv run python instruction_tuning/mlx/inference.py -m models/mlx/instruction_tuned_qwen06 -n 50

# Gradio terminal demo
uv run python instruction_tuning/app.py models/mlx/instruction_tuned_qwen06 --mlx
```

**3d. (Optional) LLM-judge the evals** — needs `OPENROUTER_API_KEY` in `.env`. The free judge model is already the default in the code.

```bash
uv run python instruction_tuning/api_inference.py -m "nvidia/nemotron-3-super-120b-a12b:free" -n 50
uv run python instruction_tuning/llm_judge.py -i instruction_tuning/evals/<your_file>_api_results.jsonl
uv run python instruction_tuning/print_judge_scores.py
```

**Where it lands:** `models/instruction_tuned_qwen06/final` (+ merged + MLX copies)

---

### Step 4 — Preference Optimization (DPO / ORPO)

Now we make the model *prefer* good answers. We generate several responses per prompt, rank them, and train it to rank the chosen one higher.

**4a. Generate multiple responses per prompt (temperature sweep):**

```bash
uv run python preference_optimization/diversity_evals/generate_responses.py \
  -m models/instruction_tuned_qwen06_merged \
  --mlx \
  --num_samples 100 --n_responses 4 --temperatures 0.3 0.5 0.7 1.0
```

> Pass `--mlx` so sampling uses the fast MLX runtime on your M2. Output is a `*_4r_temp*.jsonl` file like `pref_dataset.jsonl`.

**4b. Rank the responses** to build preference pairs (either local MLX model or the free OpenRouter API):

```bash
# Option A — local MLX ranking (no API key, free, fits 8 GB)
uv run python preference_optimization/generate_ranked_pairs.py -i pref_dataset.jsonl

# Option B — OpenRouter API (uses the free judge model; needs OPENROUTER_API_KEY)
uv run python preference_optimization/generate_ranked_pairs_openai.py -i pref_dataset.jsonl
```

Both write `<input>_ranked.jsonl` (e.g. `pref_dataset_ranked.jsonl`).

**4c. Convert ranked responses into pairwise DPO training data:**

```bash
uv run python preference_optimization/build_dpo_dataset.py -i pref_dataset_ranked.jsonl
# → pref_dataset_ranked_dpo.jsonl
```

**4d. Train DPO or ORPO on the SFT model (8 GB-safe flags):**

```bash
# DPO
uv run python preference_optimization/train_preference.py \
  --method dpo \
  --base_model_id models/instruction_tuned_qwen06/final \
  --output_model_id dpo_qwen06 \
  --dataset pref_dataset_ranked_dpo.jsonl \
  --max_seq_length 1024 \
  --batch_size 4 --grad_accum 8 --epochs 3

# ORPO
uv run python preference_optimization/train_preference.py \
  --method orpo \
  --base_model_id models/instruction_tuned_qwen06/final \
  --output_model_id orpo_qwen06 \
  --dataset pref_dataset_ranked_dpo.jsonl \
  --max_seq_length 1024 \
  --batch_size 4 --grad_accum 8 --epochs 3
```

**Where it lands:** `models/dpo_qwen06/final` (or `models/orpo_qwen06/final`)

> Optionally measure response diversity first: `uv run python preference_optimization/diversity_evals/diversity.py -i pref_dataset.jsonl`.

---

### Step 5 — Reward Model (optional but recommended)

Train a small scorer (MiniLM encoder + regression head) that rates answer quality 1–5. It pulls from the `paperbd/paper_answers_reward` dataset plus some augmented contrastive pairs.

```bash
# 5a. Train (uses dataset + augmentations, runs on MPS via your M2)
uv run python reward_models/train_reward_model.py \
  --output qwen06_v1

# → models/reward_model_finetuned_qwen06_v1  (encoder + tokenizer + head_weights.pt)
```

```bash
# 5b. Score your preference pairs with the trained reward model
uv run python preference_optimization/evaluate_reward_model.py \
  --model models/reward_model_finetuned_qwen06_v1
```

---

### Step 6 — Reasoning / RLVR (GRPO)

Teach the model to emit a hidden `think` reasoning block before the final answer with
verifiable rewards. The reasoning branch starts from the **SFT** model (not the
DPO adapter); [`cloud_train.py`](cloud_train.py) `--stages reasoning` automates
the format warm-up on the `reasoning` subset (default 10k rows, 3 epochs) before
GRPO. This is the most advanced stage.

**6a. SFT warm-up** to teach the reasoning format (8 GB-safe flags):

```bash
uv run python reasoning/sft_warmup.py \
  --base_model_id models/instruction_tuned_qwen06/final \
  --dataset paperbd/paper_instructions_300K-v1 \
  --dataset_config reasoning \
  --output_model_id reasoning_warmup_qwen06 \
  --max_seq_length 1024 \
  --batch_size 4 --grad_accum 8 --epochs 3
```

> Size this stage properly: the first cloud run used only ~168 rows / 3 optimizer
> steps and the model never learned the tags. The orchestrator now defaults to
> 10,000 rows × 3 epochs (`--reasoning-warmup-samples`,
> `--reasoning-warmup-epochs`) on the dedicated `reasoning` subset.

**6b. Edit the GRPO config** for the small model + 8 GB. Open `reasoning/grpo/config.yaml`, and at minimum change:

```yaml
model:
  name: "models/merged/instruction_tuned_qwen06"   # merged SFT base
  from_sft: false
  max_new_tokens: 512

training:
  rollout_batch_size: 8      # was 16 — smaller to fit 8 GB
  n_rollouts: 4              # was 8
  batch_size: 8              # was 32
  gradient_accumulation_steps: 4
  buffer_size: 256           # was 512
```

**6c. Run GRPO:**

```bash
uv run python reasoning/grpo/train.py reasoning/grpo/config.yaml -o reasoning_qwen06
```

> GRPO is the heaviest stage. Keep `rollout_batch_size` small and watch Activity Monitor; lower `n_rollouts` / `buffer_size` further if memory pressure spikes.

---

## 6. Environment check (CUDA boxes only)

If you're ever on an NVIDIA GPU machine, `check_env.sh` prints the 5 coordinates (torch / CUDA / C++ ABI / python / arch) that determine the correct prebuilt `flash-attention` wheel:

```bash
./check_env.sh
# or point it at the venv python:
PY=.venv/bin/python ./check_env.sh
```

---

## 7. Project layout

```
fine_tuning/
├── cpt/                        # Continued pre-training
│   ├── sft.py                  #   CPT trainer (Unsloth CUDA / HF+LoRA fallback)
│   ├── inference.py            #   Batch generation + perplexity
│   └── evals.py                #   ROUGE / BLEU / BERTScore
├── data_prep/cpt/              # arXiv download + cleaning
├── instruction_tuning/         # ChatML SFT pipeline
│   ├── sft.py                  #   SFT trainer
│   ├── inference.py            #   Batch eval (transformers)
│   ├── mlx/inference.py        #   Batch eval (MLX)
│   ├── single_inference.py     #   One-off prompt
│   ├── task_chat.py            #   Task-mode CLI chat
│   ├── app.py                  #   Gradio demo
│   ├── llm_judge.py            #   Score responses with an LLM judge
│   └── merge_adapter.py        #   Merge LoRA → full model
├── preference_optimization/    # DPO / ORPO + response ranking
│   └── diversity_evals/        #   EAD / SBERT / Vendi diversity
├── reward_models/              # Reward model training
├── reasoning/                  # Reasoning warm-up + GRPO / RLVR
│   ├── sft_warmup.py           #   Reasoning-format SFT
│   ├── env.py                  #   Rewards (think format, schema, reward model)
│   ├── eval_checkpoint.py      #   Score a checkpoint on held-out prompts
│   └── grpo/                   #   GRPO trainer + config.yaml
├── cloud_train.py              # One-command pipeline (CPT→SFT→DPO→reasoning→GRPO)
├── checkpointing.py            # Resume + Hugging Face Hub upload helpers
├── paper_researcher/           # Model-agnostic inference wrapper
├── mlx_export.py               # Merge + convert to MLX
├── check_env.sh                # CUDA wheel diagnostic
├── COLAB.md                    # Kaggle/Colab cloud walkthrough
└── pyproject.toml              # Project + dependency manifest
```

---

## 8. Notes / gotchas for the 8 GB M2

- **Unsloth is CUDA-only.** On Apple Silicon every trainer falls back to HF Transformers + PEFT. That's expected and fine.
- **Memory = mystery on the M2.** Keep `max_seq_length` low (384–1024) and use `batch_size 4 + grad_accum 8`. Watch **Activity Monitor → Memory** while training.
- **Merge before MLX.** MLX can't load LoRA adapters directly. Always run `merge_adapter.py` (or `mlx_export.py`) first.
- **`load_in_4bit` only matters on CUDA.** On Mac/CPU the full-precision (fp32/bf16) weights are loaded — another reason to use a tiny model.
- **Left padding.** Decoder-only batch generation needs `tokenizer.padding_side="left"` — already handled in `cpt/inference.py` and `instruction_tuning/inference.py`.
- **Free LLM judges are rate-limited.** The default judge model `nvidia/nemotron-3-super-120b-a12b:free` has ~20 req/min. For big batches consider a paid model or the local MLX ranker.
- **`.env` is gitignored** but `.env.example` is committed — never put your real key in `.env.example`.
- **Crashes with "Not enough memory"?** Cut `max_seq_length` in half first, then `batch_size`. GRPO: lower `rollout_batch_size` and `n_rollouts`.

---

## 9. Related resources

- [finetuning_recipes (original repo)](https://github.com/avbiswas/finetuning_recipes)
- [text-albumentations](https://github.com/avbiswas/text-albumentations) — dataset generation & augmentation
- [neural-txt](https://github.com/avbiswas/neural-txt) — inference harness & reward model
- [paper_instructions_300K-v1](https://huggingface.co/datasets/paperbd/paper_instructions_300K-v1) — instruction dataset from arXiv papers
- [neuraltxt-reward-22M](https://huggingface.co/paperbd/neuraltxt-reward-22M) — reward model used by GRPO
- [finetuning-checkpoints (HF)](https://huggingface.co/NeelContractor/finetuning-checkpoints) — checkpoint repository from this pipeline (CPT → SFT → DPO → reasoning → GRPO)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — the tiny base model used throughout

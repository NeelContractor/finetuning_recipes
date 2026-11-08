#!/usr/bin/env python3
"""Run the finetuning_recipes pipeline on a Kaggle or Colab GPU.

Pipeline
--------
    CPT -> SFT -> DPO (optional) -> reasoning warm-up -> GRPO

This is a thin orchestrator around the existing stage scripts. It does not
change their algorithms: it prepares small cloud-friendly datasets, picks
sensible sizes/epochs, and invokes the stage scripts in order.

Quick start (Colab T4 / Kaggle GPU, Internet enabled)
-----------------------------------------------------
    # From the repo root:
    python cloud_train.py --install --smoke

    # Full run on all four stages:
    python cloud_train.py --install

    # Only the stages you need:
    python cloud_train.py --stages sft grpo --install

    # Kaggle T4 x2 (DDP across both GPUs):
    python cloud_train.py --install --num-gpus 2 --smoke

    # Generated + OpenRouter-ranked preference pairs for DPO:
    OPENROUTER_API_KEY=sk-or-... python cloud_train.py \
        --stages dpo --dpo-source generate --install

Notes
-----
* Outputs go to ./models and ./data. On Colab, clone the repo under
  /content/drive/MyDrive so they survive VM restarts.
* Kaggle: turn "Internet" on in the notebook options before running --install
  (pip, Hugging Face, and arXiv all need it).
* T4/P100 are Turing/Pascal: bf16 and flash-attn are unavailable, so the stage
  scripts automatically fall back to fp16 and SDPA attention.
* `--smoke` slices datasets and runs one epoch per stage. Run it first to verify
  the environment before a long run.
* `--install` upgrades packages and then launches each stage as a fresh
  subprocess, so a runtime restart is usually unnecessary. Restart only if
  imports still fail after install.
* `--num-gpus N` launches the training stages with `accelerate launch` DDP.
  Data prep and adapter merges still run as single processes.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
STAGES = ("cpt", "sft", "dpo", "grpo")

# Installed by --install. Kept deliberately small: only what the training and
# data-prep stages in this script import. Evaluation-only packages
# (rouge-score, sacrebleu, bert-score, ...) are intentionally omitted.
CORE_PACKAGES = [
    "datasets==3.6",
    "transformers>=4.57.6,<5",
    "trl>=0.24.0",
    "peft>=0.18.1",
    "accelerate",
    "bitsandbytes",
    "arxiv",
    "pypdf",
    "requests",
    "pyyaml",
    "rich",
    "tqdm",
    "numpy",
    "pandas",
    "markdown-it-py",
    "neural-txt[hf]>=0.1.6",
    "sentence-transformers>=3.0.0",
    "openai",
    "outlines",
    "pydantic",
    "python-dotenv",
]


def log(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


def py(*args) -> list[str]:
    return [sys.executable, *[str(arg) for arg in args]]


def stage_cmd(num_gpus: int, script: str, *script_args) -> list[str]:
    """Run a training stage directly on one GPU or via Accelerate DDP."""
    if num_gpus <= 1:
        return py(script, *script_args)
    return py(
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes",
        num_gpus,
        "--mixed_precision",
        "no",
        script,
        *script_args,
    )


def run(cmd, dry_run: bool = False, env: dict | None = None) -> None:
    printable = shlex.join(str(part) for part in cmd)
    log(printable)
    if dry_run:
        return
    subprocess.run(
        [str(part) for part in cmd],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def load_env_file(path: Path) -> None:
    """Tiny .env loader so the orchestrator works before python-dotenv exists."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )
    log(f"Loaded environment from {path}")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def report_environment() -> None:
    log(f"Repository: {REPO_ROOT}")
    try:
        import torch
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        print(f"WARNING: could not import torch: {exc}")
        return

    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA is not available. Colab/Kaggle training needs a GPU "
            "runtime; continue only if you know what you are doing."
        )
        return

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU 0: {name} (sm{major}{minor}, {total_gb:.1f} GB)")
    print(f"GPUs visible: {torch.cuda.device_count()}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    print(f"torch: {torch.__version__}")
    if major < 7:
        print(
            "WARNING: Unsloth officially targets sm70+. P100/K80 may work with a "
            "very recent Unsloth but are not guaranteed. Prefer a T4 on Kaggle/Colab."
        )


def resolve_num_gpus(args: argparse.Namespace) -> int:
    """Validate --num-gpus and make sure Accelerate is available for DDP."""
    if args.num_gpus <= 1:
        return 1
    try:
        import torch
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        raise SystemExit(
            f"Could not import torch to validate --num-gpus: {exc}"
        ) from exc

    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if args.dry_run:
        log(
            f"[dry-run] would launch {args.num_gpus} DDP process(es); "
            f"{visible} GPU(s) visible"
        )
        return args.num_gpus
    if not torch.cuda.is_available() or visible < 2:
        raise SystemExit(
            f"--num-gpus {args.num_gpus} requested, but only {visible} CUDA GPU(s) "
            "are visible. On Kaggle select 'GPU T4 x2'."
        )
    if args.num_gpus > visible:
        raise SystemExit(
            f"--num-gpus {args.num_gpus} requested, but only {visible} CUDA GPU(s) "
            "are visible. Lower --num-gpus."
        )
    try:
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Multi-GPU training needs Accelerate. Run with --install or "
            "`pip install accelerate`."
        ) from exc
    return args.num_gpus


def install_deps(dry_run: bool) -> None:
    log("Installing cloud dependencies (this can take several minutes)")
    run(py("-m", "pip", "install", "--upgrade", "pip"), dry_run)
    # Unsloth first: it picks the compatible transformers/trl/datasets stack.
    run(py("-m", "pip", "install", "--upgrade", "unsloth"), dry_run)
    run(py("-m", "pip", "install", *CORE_PACKAGES), dry_run)


def _write_jsonl(dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    dataset.to_json(str(path))


def _load_hf_dataset(name: str, split: str, config: str | None = None):
    from datasets import load_dataset

    if config is None:
        return load_dataset(name, split=split)
    return load_dataset(name, config, split=split)


def prepare_cpt_data(args: argparse.Namespace, dry_run: bool) -> tuple[Path, Path]:
    """Download arXiv papers and build cpt_train.jsonl / cpt_val.jsonl."""
    papers_dir = REPO_ROOT / "data_prep" / "cpt" / "papers"
    train_path = DATA_DIR / "cpt_train.jsonl"
    val_path = DATA_DIR / "cpt_val.jsonl"

    if dry_run:
        log(f"[dry-run] would prepare CPT data from {papers_dir}")
        return train_path, val_path

    if not args.force and train_path.exists() and val_path.exists():
        log(f"Reusing {train_path.name} + {val_path.name}")
        return train_path, val_path

    if args.cpt_papers < 55:
        raise SystemExit(
            "get_dataset.py reserves 50 papers for validation, so --cpt-papers "
            "must be at least 55 (recommended: 100+)."
        )

    if args.force and papers_dir.exists():
        log(f"Removing existing papers at {papers_dir}")
        shutil.rmtree(papers_dir)

    existing = len(list(papers_dir.rglob("*.txt"))) if papers_dir.exists() else 0
    missing = max(0, args.cpt_papers - existing)
    if missing:
        log(f"Downloading {missing} arXiv papers into {papers_dir}")
        run(
            py(
                "data_prep/cpt/download_arxiv_batch.py",
                "--output-dir",
                papers_dir,
                "--target-count",
                missing,
                "--workers",
                5,
            ),
            dry_run,
        )
    else:
        log(f"Reusing {existing} papers in {papers_dir}")

    total = len(list(papers_dir.rglob("*.txt")))
    if total < 55:
        raise SystemExit(
            f"Only {total} papers available after download; need at least 55."
        )

    # get_dataset.py appends, so remove any previous output first.
    for old in REPO_ROOT.glob("cpt_*dataset_*.jsonl"):
        old.unlink()

    run(py("data_prep/cpt/get_dataset.py", papers_dir), dry_run)

    train_files = sorted(REPO_ROOT.glob("cpt_train_dataset_*.jsonl"))
    val_files = sorted(REPO_ROOT.glob("cpt_val_dataset_*.jsonl"))
    if not train_files or not val_files:
        raise SystemExit("get_dataset.py did not produce the expected JSONL files.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if train_path.exists():
        train_path.unlink()
    if val_path.exists():
        val_path.unlink()
    shutil.move(str(train_files[0]), train_path)
    shutil.move(str(val_files[0]), val_path)
    log(f"CPT data: {train_path.name} + {val_path.name}")
    return train_path, val_path


def prepare_sft_data(args: argparse.Namespace, dry_run: bool) -> Path:
    """Subsample the instruction dataset into data/sft/train.jsonl."""
    sft_dir = DATA_DIR / "sft"
    train_path = sft_dir / "train.jsonl"

    if not args.force and train_path.exists():
        log(f"Reusing {train_path}")
        return sft_dir
    if dry_run:
        log(f"[dry-run] would write {args.sft_samples} SFT rows to {train_path}")
        return sft_dir

    dataset = _load_hf_dataset(args.sft_dataset, split="train")
    dataset = dataset.shuffle(seed=args.seed).select(
        range(min(args.sft_samples, len(dataset)))
    )
    _write_jsonl(dataset, train_path)
    log(f"SFT data: {len(dataset)} rows -> {train_path}")
    return sft_dir


def prepare_dpo_data(args: argparse.Namespace, dry_run: bool) -> Path:
    """Subsample the Hugging Face preference dataset into data/dpo/train.jsonl."""
    dpo_dir = DATA_DIR / "dpo"
    train_path = dpo_dir / "train.jsonl"

    if not args.force and train_path.exists():
        log(f"Reusing {train_path}")
        return dpo_dir
    if dry_run:
        log(f"[dry-run] would write {args.dpo_samples} DPO rows to {train_path}")
        return dpo_dir

    dataset = _load_hf_dataset(args.dpo_dataset, split="train")
    dataset = dataset.shuffle(seed=args.seed).select(
        range(min(args.dpo_samples, len(dataset)))
    )
    _write_jsonl(dataset, train_path)
    log(f"DPO data: {len(dataset)} rows -> {train_path}")
    return dpo_dir


def prepare_reasoning_data(args: argparse.Namespace, dry_run: bool) -> Path:
    """Build reasoning train/test JSONL for the GRPO warm-up and eval."""
    reason_dir = DATA_DIR / "reasoning"
    train_path = reason_dir / "train.jsonl"
    test_path = reason_dir / "test.jsonl"

    if not args.force and train_path.exists() and test_path.exists():
        log(f"Reusing {reason_dir}")
        return reason_dir
    if dry_run:
        log(f"[dry-run] would write reasoning data to {reason_dir}")
        return reason_dir

    train_ds = _load_hf_dataset(args.sft_dataset, split="train", config="reasoning")
    train_ds = train_ds.shuffle(seed=args.seed).select(
        range(min(args.grpo_samples, len(train_ds)))
    )
    _write_jsonl(train_ds, train_path)

    test_ds = _load_hf_dataset(args.sft_dataset, split="test")
    test_ds = test_ds.shuffle(seed=args.seed).select(
        range(min(args.grpo_eval_samples, len(test_ds)))
    )
    _write_jsonl(test_ds, test_path)
    log(
        f"Reasoning data: train={len(train_ds)} test={len(test_ds)} "
        f"-> {reason_dir}"
    )
    return reason_dir


def require_openrouter() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "--dpo-source generate with --dpo-ranker openrouter needs "
            "OPENROUTER_API_KEY. Pass --openrouter-key, put it in .env, or "
            "use --dpo-ranker local."
        )


def resolve_dpo_ranker(args: argparse.Namespace) -> str:
    if args.dpo_ranker != "auto":
        return args.dpo_ranker
    return "openrouter" if os.environ.get("OPENROUTER_API_KEY") else "local"


def checkpoint_cmd_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if args.resume:
        extra.append("--resume")
    if args.push_to_hub:
        extra.extend(
            [
                "--push-to-hub",
                "--hub-repo",
                args.hub_repo,
                "--hub-upload-every-saves",
                str(args.hub_upload_every_saves),
            ]
        )
        if not args.hub_private:
            extra.append("--no-hub-private")
    return extra


def stage_cpt(args: argparse.Namespace, dry_run: bool, env: dict) -> Path:
    train_path, val_path = prepare_cpt_data(args, dry_run)
    run(
        stage_cmd(
            args.num_gpus,
            "cpt/sft.py",
            "--base_model_id",
            args.model,
            "--output_model_id",
            "cpt_qwen06",
            "--dataset_path",
            train_path,
            "--test_dataset_path",
            val_path,
            "--max_seq_length",
            args.cpt_max_seq_length,
            "--batch_size",
            args.cpt_batch_size,
            "--epochs",
            args.cpt_epochs,
            *checkpoint_cmd_args(args),
        ),
        dry_run,
        env,
    )
    adapter = MODELS_DIR / "cpt_qwen06" / "final"
    if args.no_merge:
        return adapter

    # Merge CPT before SFT so the SFT adapter hangs off a full model. This also
    # keeps merge_adapter.py's "read base from adapter_config.json" working.
    merged = MODELS_DIR / "cpt_qwen06_merged"
    run(
        py(
            "instruction_tuning/merge_adapter.py",
            "--adapter_path",
            adapter,
            "--output_path",
            merged,
        ),
        dry_run,
        env,
    )
    return merged


def stage_sft(
    args: argparse.Namespace, dry_run: bool, env: dict, base_model: Path
) -> Path:
    data_dir = prepare_sft_data(args, dry_run)
    run(
        stage_cmd(
            args.num_gpus,
            "instruction_tuning/sft.py",
            "--base_model_id",
            base_model,
            "--output_model_id",
            "sft_qwen06",
            "--dataset",
            data_dir,
            "--max_seq_length",
            args.sft_max_seq_length,
            "--batch_size",
            args.sft_batch_size,
            "--grad_accum",
            args.sft_grad_accum,
            "--epochs",
            args.sft_epochs,
            *checkpoint_cmd_args(args),
        ),
        dry_run,
        env,
    )
    adapter = MODELS_DIR / "sft_qwen06" / "final"
    if args.no_merge:
        return adapter

    merged = MODELS_DIR / "sft_qwen06_merged"
    run(
        py(
            "instruction_tuning/merge_adapter.py",
            "--adapter_path",
            adapter,
            "--output_path",
            merged,
        ),
        dry_run,
        env,
    )
    return merged


def stage_dpo(
    args: argparse.Namespace, dry_run: bool, env: dict, base_model: Path
) -> Path:
    if args.dpo_source == "generate":
        ranker = resolve_dpo_ranker(args)
        if ranker == "openrouter":
            require_openrouter()
        merged = MODELS_DIR / "sft_qwen06_merged"
        if not dry_run and not merged.exists():
            raise SystemExit(
                "DPO generation needs models/sft_qwen06_merged. Run the SFT "
                "stage without --no-merge first."
            )

        raw_path = DATA_DIR / "dpo_generated.jsonl"
        run(
            py(
                "preference_optimization/diversity_evals/generate_responses.py",
                "--model_path",
                merged,
                "--num_samples",
                args.dpo_gen_samples,
                "--n_responses",
                4,
                "--temperatures",
                0.3,
                0.5,
                0.7,
                1.0,
                "--max_new_tokens",
                256,
                "--batch_size",
                2,
                "--output_file",
                raw_path,
            ),
            dry_run,
            env,
        )
        if ranker == "openrouter":
            ranker_cmd = py(
                "preference_optimization/generate_ranked_pairs_openai.py",
                "-i",
                raw_path,
            )
        else:
            ranker_cmd = py(
                "preference_optimization/generate_ranked_pairs.py",
                "-i",
                raw_path,
                "--backend",
                "auto",
            )
            if args.dpo_ranker_model:
                ranker_cmd += ["--model_path", args.dpo_ranker_model]
            if args.dpo_ranker_load_in_4bit:
                ranker_cmd += ["--load_in_4bit"]
        run(ranker_cmd, dry_run, env)
        ranked_path = DATA_DIR / "dpo_generated_ranked.jsonl"
        dpo_dir = DATA_DIR / "dpo"
        dpo_dir.mkdir(parents=True, exist_ok=True)
        run(
            py(
                "preference_optimization/build_dpo_dataset.py",
                "-i",
                ranked_path,
                "-o",
                dpo_dir / "train.jsonl",
            ),
            dry_run,
            env,
        )
    else:
        prepare_dpo_data(args, dry_run)
        dpo_dir = DATA_DIR / "dpo"

    run(
        stage_cmd(
            args.num_gpus,
            "preference_optimization/train_preference.py",
            "--method",
            "dpo",
            "--base_model_id",
            base_model,
            "--output_model_id",
            "dpo_qwen06",
            "--dataset",
            dpo_dir,
            "--max_seq_length",
            args.dpo_max_seq_length,
            "--max_prompt_length",
            args.dpo_max_prompt_length,
            "--batch_size",
            args.dpo_batch_size,
            "--grad_accum",
            args.dpo_grad_accum,
            "--epochs",
            args.dpo_epochs,
            *checkpoint_cmd_args(args),
        ),
        dry_run,
        env,
    )
    return MODELS_DIR / "dpo_qwen06" / "final"


def write_grpo_config(
    path: Path,
    args: argparse.Namespace,
    reasoning_dir: Path,
    model_path: Path,
    from_sft: bool,
) -> None:
    import yaml

    # dr_grpo's denominator is a loss-scale constant (~mean completion length),
    # NOT the generation cap. Never tie it to max_new_tokens.
    loss_max_tokens = min(args.grpo_loss_max_tokens, args.grpo_max_new_tokens)
    config = {
        "model": {
            "name": str(model_path),
            "from_sft": from_sft,
            "max_new_tokens": args.grpo_max_new_tokens,
        },
        "training": {
            "rollout_batch_size": args.grpo_rollout_batch_size,
            "n_rollouts": args.grpo_n_rollouts,
            # 0.5 gave higher mean reward and fewer zero-std groups than 0.8.
            "temperature": 0.5,
            "eval_temperature": 0.2,
            "batch_size": args.grpo_batch_size,
            # Cap sequences per forward/backward so the policy+ref full-vocab
            # logits fit on a 14 GB T4 even when batch_size > 1.
            "micro_batch_size": 1,
            "gradient_accumulation_steps": args.grpo_grad_accum,
            "learning_rate": args.grpo_lr,
            "num_epochs": args.grpo_epochs,
            "log_every": 1,
            "eval_every_train_step": args.grpo_eval_every,
            "top_p": 0.95,
            "num_repeats": args.grpo_num_repeats,
            "buffer_size": args.grpo_buffer_size,
            "std_normalize_advantages": False,
        },
        "data": {
            "dataset": str(reasoning_dir),
            "train_split": "train",
            "test_split": "test",
            "train_data_size": args.grpo_samples,
            "eval_data_size": args.grpo_eval_samples,
            "test_batch_size": args.grpo_eval_batch_size,
            "max_prompt_length": max(
                128, args.grpo_max_seq_length - args.grpo_max_new_tokens
            ),
        },
        "loss": {
            "kld_weight": 0.02,
            "entropy_weight": 0.0,
            "loss_implementation": "dr_grpo",
            "max_tokens": loss_max_tokens,
        },
        "reward": {
            "require_think_format": args.grpo_require_think_format,
            "include_format_rewards": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    log(f"GRPO config: {path}")


def stage_grpo(
    args: argparse.Namespace, dry_run: bool, env: dict, base_model: Path
) -> Path:
    reasoning_dir = prepare_reasoning_data(args, dry_run)

    run(
        stage_cmd(
            args.num_gpus,
            "reasoning/sft_warmup.py",
            "--base_model_id",
            base_model,
            "--dataset",
            reasoning_dir,
            "--dataset_config",
            "default",
            "--output_model_id",
            "reasoning_warmup_qwen06",
            "--max_seq_length",
            args.grpo_max_seq_length,
            "--batch_size",
            args.grpo_sft_batch_size,
            "--grad_accum",
            args.grpo_grad_accum,
            "--epochs",
            args.grpo_warmup_epochs,
            *checkpoint_cmd_args(args),
        ),
        dry_run,
        env,
    )

    warmup_adapter = MODELS_DIR / "reasoning_warmup_qwen06" / "final"
    if args.no_merge:
        grpo_model = warmup_adapter
        from_sft = True
    else:
        grpo_model = MODELS_DIR / "reasoning_warmup_qwen06_merged"
        run(
            py(
                "instruction_tuning/merge_adapter.py",
                "--adapter_path",
                warmup_adapter,
                "--output_path",
                grpo_model,
            ),
            dry_run,
            env,
        )
        from_sft = False

    config_path = DATA_DIR / "grpo_cloud_config.yaml"
    if dry_run:
        log(f"[dry-run] would write GRPO config to {config_path}")
    else:
        write_grpo_config(config_path, args, reasoning_dir, grpo_model, from_sft)

    run(
        stage_cmd(
            args.num_gpus,
            "reasoning/grpo/train.py",
            config_path,
            "-o",
            "grpo_qwen06",
            *checkpoint_cmd_args(args),
        ),
        dry_run,
        env,
    )
    return MODELS_DIR / "grpo_qwen06"


def apply_smoke(args: argparse.Namespace) -> None:
    args.cpt_papers = min(args.cpt_papers, 60)
    args.cpt_epochs = min(args.cpt_epochs, 1)

    args.sft_samples = min(args.sft_samples, 2000)
    args.sft_epochs = min(args.sft_epochs, 1)
    args.sft_max_seq_length = min(args.sft_max_seq_length, 512)

    args.dpo_samples = min(args.dpo_samples, 2000)
    args.dpo_gen_samples = min(args.dpo_gen_samples, 10)
    args.dpo_epochs = min(args.dpo_epochs, 1)
    args.dpo_max_seq_length = min(args.dpo_max_seq_length, 512)
    args.dpo_max_prompt_length = min(args.dpo_max_prompt_length, 384)

    args.grpo_samples = min(args.grpo_samples, 32)
    args.grpo_eval_samples = min(args.grpo_eval_samples, 8)
    args.grpo_epochs = min(args.grpo_epochs, 1)
    args.grpo_warmup_epochs = min(args.grpo_warmup_epochs, 1)
    args.grpo_max_seq_length = min(args.grpo_max_seq_length, 2048)
    args.grpo_rollout_batch_size = min(args.grpo_rollout_batch_size, 1)
    args.grpo_n_rollouts = max(2, min(args.grpo_n_rollouts, 2))
    args.grpo_batch_size = min(args.grpo_batch_size, 1)
    args.grpo_buffer_size = min(args.grpo_buffer_size, 8)
    args.grpo_max_new_tokens = min(args.grpo_max_new_tokens, 64)
    args.grpo_loss_max_tokens = min(args.grpo_loss_max_tokens, 64)
    args.grpo_num_repeats = min(args.grpo_num_repeats, 1)
    args.grpo_eval_every = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPT/SFT/DPO/GRPO on a Colab or Kaggle GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="Subset of stages to run (always executed in pipeline order).",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs for DDP training. Use 2 on Kaggle 'GPU T4 x2'.",
    )
    parser.add_argument(
        "--install", action="store_true", help="pip install deps first."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny dataset slices + 1 epoch per stage.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download / regenerate prepared datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Keep LoRA adapters separate instead of merging (debugging only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each training stage from its latest local checkpoint.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload checkpoints and final models to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hub-repo",
        default=os.environ.get("HF_CHECKPOINT_REPO"),
        help=(
            "HF repo id for checkpoints, e.g. user/project-checkpoints. "
            "Defaults to HF_CHECKPOINT_REPO."
        ),
    )
    parser.add_argument(
        "--hub-private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create the HF checkpoint repo as private.",
    )
    parser.add_argument(
        "--hub-upload-every-saves",
        type=int,
        default=1,
        metavar="N",
        help="Upload every Nth Trainer save (1 = every save).",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--openrouter-key", default=None)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument(
        "--sft-dataset",
        default="paperbd/paper_instructions_300K-v1",
    )
    parser.add_argument(
        "--dpo-dataset",
        default="paperbd/paper_preference_150K-v1",
    )
    parser.add_argument(
        "--dpo-source",
        choices=["hf", "generate"],
        default="hf",
        help="Use the existing HF preference dataset or generate+rank pairs.",
    )
    parser.add_argument(
        "--dpo-ranker",
        choices=["auto", "local", "openrouter"],
        default="auto",
        help=(
            "How to rank generated DPO pairs. auto uses OpenRouter when "
            "OPENROUTER_API_KEY is set, otherwise a local Transformers model."
        ),
    )
    parser.add_argument(
        "--dpo-ranker-model",
        default=None,
        help=(
            "Local HF id/path for --dpo-ranker local. Defaults to "
            "Qwen/Qwen3-0.6B; use Qwen/Qwen2.5-3B-Instruct for better rankings."
        ),
    )
    parser.add_argument(
        "--dpo-ranker-load-in-4bit",
        action="store_true",
        help="Load the local DPO ranker in 4-bit (requires bitsandbytes).",
    )

    parser.add_argument("--cpt-papers", type=int, default=120)
    parser.add_argument("--cpt-epochs", type=int, default=1)
    parser.add_argument("--cpt-max-seq-length", type=int, default=512)
    parser.add_argument("--cpt-batch-size", type=int, default=8)

    parser.add_argument("--sft-samples", type=int, default=20000)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--sft-max-seq-length", type=int, default=1024)
    parser.add_argument("--sft-batch-size", type=int, default=4)
    parser.add_argument("--sft-grad-accum", type=int, default=8)

    parser.add_argument("--dpo-samples", type=int, default=5000)
    parser.add_argument("--dpo-gen-samples", type=int, default=50)
    parser.add_argument("--dpo-epochs", type=int, default=1)
    parser.add_argument("--dpo-max-seq-length", type=int, default=1024)
    parser.add_argument("--dpo-max-prompt-length", type=int, default=768)
    parser.add_argument("--dpo-batch-size", type=int, default=4)
    parser.add_argument("--dpo-grad-accum", type=int, default=8)

    parser.add_argument("--grpo-samples", type=int, default=256)
    parser.add_argument("--grpo-eval-samples", type=int, default=64)
    parser.add_argument("--grpo-warmup-epochs", type=int, default=1)
    parser.add_argument("--grpo-epochs", type=int, default=1)
    parser.add_argument("--grpo-max-seq-length", type=int, default=1024)
    parser.add_argument("--grpo-sft-batch-size", type=int, default=4)
    parser.add_argument("--grpo-grad-accum", type=int, default=8)
    parser.add_argument("--grpo-rollout-batch-size", type=int, default=4)
    parser.add_argument("--grpo-n-rollouts", type=int, default=4)
    parser.add_argument("--grpo-batch-size", type=int, default=4)
    parser.add_argument("--grpo-buffer-size", type=int, default=64)
    parser.add_argument("--grpo-max-new-tokens", type=int, default=384)
    parser.add_argument(
        "--grpo-loss-max-tokens",
        type=int,
        default=256,
        help="dr_grpo loss denominator (~mean completion length), not a gen cap.",
    )
    parser.add_argument("--grpo-num-repeats", type=int, default=3)
    parser.add_argument("--grpo-lr", type=float, default=5e-6)
    parser.add_argument("--grpo-eval-every", type=int, default=5)
    parser.add_argument("--grpo-eval-batch-size", type=int, default=4)
    parser.add_argument(
        "--grpo-require-think-format",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Gate the semantic reward on a valid  thinking...</think> block. "
            "Default off so a model that has not yet learned the format still "
            "gets a non-zero (semantic) reward and non-zero GRPO advantages."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke(args)

    os.chdir(REPO_ROOT)
    ensure_dirs()
    load_env_file(REPO_ROOT / args.env_file)
    args.hub_repo = args.hub_repo or os.environ.get("HF_CHECKPOINT_REPO")
    if args.push_to_hub and not args.hub_repo:
        raise SystemExit(
            "--push-to-hub requires --hub-repo (or HF_CHECKPOINT_REPO) "
            "with a repo id such as user/project-checkpoints."
        )
    if args.openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_key

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if (
        args.dpo_source == "generate"
        and "dpo" in args.stages
        and resolve_dpo_ranker(args) == "openrouter"
    ):
        require_openrouter()

    if args.install:
        install_deps(args.dry_run)
    report_environment()
    args.num_gpus = resolve_num_gpus(args)
    if args.num_gpus > 1:
        log(f"DDP enabled: {args.num_gpus} processes per training stage")

    selected = args.stages
    log("Stages: " + ", ".join(selected))

    cpt_final = MODELS_DIR / "cpt_qwen06" / "final"
    cpt_merged = MODELS_DIR / "cpt_qwen06_merged"
    sft_final = MODELS_DIR / "sft_qwen06" / "final"
    sft_merged = MODELS_DIR / "sft_qwen06_merged"

    cpt_output: Path | None = None
    if "cpt" in selected:
        cpt_output = stage_cpt(args, args.dry_run, env)

    sft_output: Path | None = None
    if "sft" in selected:
        if cpt_output is not None:
            sft_base = cpt_output
        elif cpt_merged.exists():
            sft_base = cpt_merged
        elif cpt_final.exists():
            sft_base = cpt_final
        else:
            sft_base = Path(args.model)
        sft_output = stage_sft(args, args.dry_run, env, sft_base)

    def resolve_sft_checkpoint(stage: str) -> Path:
        if sft_output is not None:
            return sft_output
        if sft_merged.exists():
            return sft_merged
        if sft_final.exists():
            if args.no_merge:
                return sft_final
            # An adapter exists but no merged copy: merge it so the next stage
            # starts from a full model (nested adapters are fragile).
            run(
                py(
                    "instruction_tuning/merge_adapter.py",
                    "--adapter_path",
                    sft_final,
                    "--output_path",
                    sft_merged,
                ),
                args.dry_run,
                env,
            )
            return sft_merged
        raise SystemExit(
            f"{stage} needs an SFT checkpoint. Run --stages sft {stage.lower()}, "
            "or place one at models/sft_qwen06_merged or models/sft_qwen06/final."
        )

    if "dpo" in selected:
        stage_dpo(args, args.dry_run, env, resolve_sft_checkpoint("DPO"))

    if "grpo" in selected:
        stage_grpo(args, args.dry_run, env, resolve_sft_checkpoint("GRPO"))

    log("Done. Checkpoints are under ./models")


if __name__ == "__main__":
    main()

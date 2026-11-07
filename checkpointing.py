"""Checkpoint resume and optional Hugging Face Hub upload helpers.

The training stages use different checkpoint layouts:
  * HF Trainer stages write ``models/<run>/checkpoint-<step>/trainer_state.json``
  * GRPO writes ``models/<run>/llm`` and ``models/<run>/llm_best_<n>``

This module keeps the resume and upload logic in one place so every stage
behaves the same way.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the latest checkpoint in the output directory. "
            "If none exists, start fresh."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Explicit checkpoint path. Overrides --resume.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload checkpoints and the final model to Hugging Face Hub.",
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


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the latest HF Trainer ``checkpoint-*`` directory, if any."""
    root = Path(output_dir)
    if not root.exists():
        return None

    latest: tuple[int, Path] | None = None
    for path in root.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        if not (path / "trainer_state.json").exists():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if latest is None or step > latest[0]:
            latest = (step, path)
    return latest[1] if latest else None


def find_latest_model_dir(output_dir: str | Path) -> Path | None:
    """Return the most recently written GRPO ``llm*`` directory."""
    root = Path(output_dir)
    if not root.exists():
        return None

    candidates = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if path.name != "llm" and not path.name.startswith("llm_best_"):
            continue
        if (path / "config.json").exists() or (path / "adapter_config.json").exists():
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_resume_checkpoint(
    output_dir: str | Path,
    explicit: str | None = None,
    auto: bool = False,
    finder=find_latest_checkpoint,
) -> str | None:
    """Resolve an explicit checkpoint path or auto-discover the latest one."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise SystemExit(f"--resume_from_checkpoint does not exist: {path}")
        print(f"[resume] Using explicit checkpoint: {path}")
        return str(path)

    if auto:
        path = finder(output_dir)
        if path is None:
            print(f"[resume] No checkpoint found in {output_dir}; starting fresh.")
            return None
        print(f"[resume] Resuming from {path}")
        return str(path)

    return None


def hub_repo_from_args(args: argparse.Namespace) -> str | None:
    if not getattr(args, "push_to_hub", False):
        return None
    repo_id = getattr(args, "hub_repo", None)
    if not repo_id:
        raise SystemExit(
            "--push-to-hub requires --hub-repo (or HF_CHECKPOINT_REPO) "
            "with a repo id such as user/project-checkpoints."
        )
    return repo_id


def push_folder_to_hub(
    folder_path: str | Path,
    repo_id: str,
    path_in_repo: str | None = None,
    private: bool = True,
    token: str | None = None,
    commit_message: str | None = None,
) -> None:
    """Upload a local checkpoint/model folder to a Hugging Face repo."""
    from huggingface_hub import HfApi

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Cannot upload missing folder: {folder}")

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    target = path_in_repo or folder.name
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        path_in_repo=target,
        commit_message=commit_message or f"Upload {target}",
    )
    print(f"[hub] Uploaded {folder} -> {repo_id}/{target}")


def make_hub_checkpoint_callback(
    repo_id: str,
    path_prefix: str,
    private: bool = True,
    upload_every: int = 1,
    token: str | None = None,
):
    """Create a TrainerCallback that uploads each saved checkpoint."""
    from transformers import TrainerCallback

    class HubCheckpointCallback(TrainerCallback):
        def __init__(self):
            self.save_count = 0

        def on_save(self, args, state, control, **kwargs):
            if not getattr(state, "is_world_process_zero", True):
                return control

            self.save_count += 1
            if upload_every > 1 and self.save_count % upload_every != 0:
                return control

            checkpoint = kwargs.get("checkpoint") or getattr(state, "save_path", None)
            if not checkpoint:
                return control

            target = f"{path_prefix}/{Path(checkpoint).name}".strip("/")
            try:
                push_folder_to_hub(
                    checkpoint,
                    repo_id,
                    path_in_repo=target,
                    private=private,
                    token=token,
                    commit_message=f"Upload {target}",
                )
            except Exception as exc:
                print(f"[hub] WARNING: failed to upload {checkpoint}: {exc}")
            return control

    return HubCheckpointCallback()


def upload_final_model(
    output_dir: str | Path,
    repo_id: str,
    output_model_id: str,
    private: bool = True,
    token: str | None = None,
) -> None:
    final_dir = Path(output_dir) / "final"
    if not final_dir.exists():
        return
    push_folder_to_hub(
        final_dir,
        repo_id,
        path_in_repo=f"{output_model_id}/final",
        private=private,
        token=token,
        commit_message=f"Upload {output_model_id}/final",
    )

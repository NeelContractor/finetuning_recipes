from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

DEFAULT_MLX_MODEL = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_TRANSFORMERS_MODEL = "Qwen/Qwen3-0.6B"

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
"""

RANKING_PROMPT = """\
You are an expert evaluator for AI assistant responses on research and academic questions.

Given a question and {n} candidate responses, rank the responses from **best** (1st) to **worst** ({n}th) based on overall quality: correctness, relevance, completeness, and clarity.

Question:
{question}

{responses_block}

Provide a brief reasoning for your ranking, then output the ranking as a JSON object with:
- reasoning: str  (1-2 sentences)
- ranking_list: list[int]  (response indices 0-{n_minus_1} ordered from best to worst; every index must appear exactly once)
"""


class RankingOutput(BaseModel):
    reasoning: str = Field(..., description="Brief explanation for the ranking.")
    ranking_list: list[int] = Field(
        ...,
        description="Permutation of [0..n-1] ordered from best to worst.",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate ranked pairs for preference data with a local model. "
            "Uses MLX on Apple Silicon and Transformers/Outlines on CUDA (Kaggle)."
        )
    )
    p.add_argument(
        "--input_file",
        "-i",
        type=str,
        default="preference_optimization/evals/pref_dataset/train_4r_temp0.5.jsonl",
    )
    p.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Defaults to <input>_ranked.jsonl",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "mlx", "transformers"],
        default="auto",
        help="auto uses MLX on Apple Silicon, otherwise Transformers.",
    )
    p.add_argument(
        "--model_path",
        "-m",
        type=str,
        default=None,
        help=(
            "Local path or HF id. Defaults to mlx-community/Qwen3-0.6B-4bit "
            "for MLX and Qwen/Qwen3-0.6B for Transformers."
        ),
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="4-bit quantization for Transformers backends (requires bitsandbytes).",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--limit", "-n", type=int, default=None)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_retries", type=int, default=2)
    return p.parse_args()


def count_valid_lines(path: Path) -> int:
    if not path.exists():
        return 0
    valid = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json.loads(line)
                valid += 1
            except json.JSONDecodeError:
                break
    return valid


def build_ranking_prompt_text(question: str, responses: list[str]) -> str:
    n = len(responses)
    responses_block = "\n\n".join(
        f"Response {i}:\n{r}" for i, r in enumerate(responses)
    )
    return RANKING_PROMPT.format(
        question=question,
        n=n,
        n_minus_1=n - 1,
        responses_block=responses_block,
    )


def validate_ranking(ranking: list[int], expected_len: int) -> bool:
    if len(ranking) != expected_len:
        return False
    return set(ranking) == set(range(expected_len))


def _extract_result(raw: Any) -> dict[str, Any] | None:
    """Normalise batch_chat output to a plain dict."""
    if raw is None:
        return None
    if hasattr(raw, "model_dump"):
        return raw.model_dump()
    if isinstance(raw, dict):
        return raw
    # Some older versions wrap text in an object with a .text attribute
    if hasattr(raw, "text"):
        try:
            return json.loads(raw.text)
        except json.JSONDecodeError:
            return None
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return None


def resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx_vlm_batch_outlines  # noqa: F401

            return "mlx"
        except ImportError:
            pass
    return "transformers"


def load_mlx_backend(model_path: str) -> Callable:
    try:
        from mlx_vlm_batch_outlines import batch_chat, load
    except ImportError as exc:
        raise SystemExit(
            "MLX backend requested but mlx-vlm-batch-outlines is not installed. "
            "On Kaggle/CUDA use --backend transformers instead."
        ) from exc

    model, processor = load(model_path)

    def generate(conversations, output_type, max_tokens):
        return batch_chat(
            model,
            processor,
            conversations,
            output_type=output_type,
            max_tokens=max_tokens,
        )

    return generate


def load_transformers_backend(
    model_path: str,
    device: str,
    dtype: str,
    load_in_4bit: bool,
    trust_remote_code: bool,
) -> Callable:
    try:
        import outlines
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Transformers backend needs outlines, transformers, torch, and accelerate. "
            "Install them with: pip install outlines transformers accelerate"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device,
        "low_cpu_mem_usage": True,
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise SystemExit(
                "--load_in_4bit needs bitsandbytes. Install it with: pip install bitsandbytes"
            ) from exc
        compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype == "auto":
            dtype = "float16" if torch.cuda.is_available() else "float32"
        model_kwargs["torch_dtype"] = dtype_map[dtype]

    hf_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    hf_model.eval()
    model = outlines.from_transformers(hf_model, tokenizer)

    def format_prompt(conversation: list[dict[str, str]]) -> str:
        try:
            return tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            rendered = "\n\n".join(
                f"{message['role'].capitalize()}: {message['content']}"
                for message in conversation
            )
            return rendered + "\n\nAssistant:"

    def generate(conversations, output_type, max_tokens):
        results = []
        for conversation in conversations:
            prompt = format_prompt(conversation)
            results.append(
                model(prompt, output_type, max_new_tokens=max_tokens)
            )
        return results

    return generate


def load_backend(args: argparse.Namespace) -> tuple[str, str, Callable]:
    backend = resolve_backend(args.backend)
    default_model = (
        DEFAULT_MLX_MODEL if backend == "mlx" else DEFAULT_TRANSFORMERS_MODEL
    )
    model_path = args.model_path or default_model

    print(f"Loading {backend} model from {model_path} …")
    if backend == "mlx":
        generate = load_mlx_backend(model_path)
    else:
        generate = load_transformers_backend(
            model_path,
            device=args.device,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            trust_remote_code=args.trust_remote_code,
        )
    print("Model loaded.\n")
    return backend, model_path, generate


def main() -> None:
    args = parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace(".jsonl", "_ranked.jsonl")
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend, model_path, generate_batch = load_backend(args)
    print(f"Backend: {backend} | model: {model_path}\n")

    with open(args.input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if args.limit is not None:
        records = records[: args.limit]

    existing = count_valid_lines(out_path)
    if existing > 0:
        print(f"Resuming: {existing}/{len(records)} records already in output.")
    records = records[existing:]

    total = len(records)
    print(f"Processing {total} records in batches of {args.batch_size} …")

    t0 = time.time()
    written = existing

    for batch_start in range(0, total, args.batch_size):
        batch = records[batch_start : batch_start + args.batch_size]
        batch_size = len(batch)

        # Build text-only conversations for this batch
        conversations: list[list[dict[str, str]]] = []
        for rec in batch:
            user_text = build_ranking_prompt_text(rec["question"], rec["responses"])
            conversations.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ]
            )

        # --- Batch generation ------------------------------------------------
        batch_results: list[Any] = [None] * batch_size
        try:
            raw_batch = generate_batch(
                conversations,
                RankingOutput,
                args.max_tokens,
            )
            # Normalise to a plain list
            if batch_size == 1 and not isinstance(raw_batch, list):
                batch_results = [raw_batch]
            elif isinstance(raw_batch, list):
                batch_results = raw_batch
            else:
                # The library may return a list-like wrapper; try iterating
                batch_results = list(raw_batch)
        except Exception as e:
            print(f"  Batch {batch_start // args.batch_size + 1} failed: {e}")

        # --- Write / retry per record ----------------------------------------
        for idx_in_batch, (rec, raw) in enumerate(zip(batch, batch_results)):
            global_idx = existing + batch_start + idx_in_batch
            n_resp = len(rec["responses"])

            ranking: list[int] | None = None
            reasoning = ""

            # 1. Try batch result
            data = _extract_result(raw)
            if data is not None:
                candidate = data.get("ranking_list", [])
                if validate_ranking(candidate, n_resp):
                    ranking = candidate
                    reasoning = data.get("reasoning", "")

            # 2. Retry individually if batch result was missing / invalid
            for attempt in range(args.max_retries):
                if ranking is not None:
                    break
                try:
                    single_raw = generate_batch(
                        [
                            [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": build_ranking_prompt_text(
                                        rec["question"], rec["responses"]
                                    ),
                                },
                            ]
                        ],
                        RankingOutput,
                        args.max_tokens,
                    )
                    # normalise single-item wrapper
                    if not isinstance(single_raw, list):
                        single_raw = [single_raw]
                    data = _extract_result(single_raw[0]) if single_raw else None
                    if data is not None:
                        candidate = data.get("ranking_list", [])
                        if validate_ranking(candidate, n_resp):
                            ranking = candidate
                            reasoning = data.get("reasoning", "")
                            break
                        else:
                            print(
                                f"  [{global_idx+1}] attempt {attempt+1} invalid ranking: {candidate}"
                            )
                except Exception as e:
                    print(f"  [{global_idx+1}] attempt {attempt+1} error: {e}")

            if ranking is None:
                print(f"  [{global_idx+1}] WARNING: all retries failed, using fallback ranking.")
                ranking = list(range(n_resp))
                reasoning = "Fallback ranking due to generation failure."

            out_rec = {
                **rec,
                "ranking_list": ranking,
                "ranking_reasoning": reasoning,
            }
            with open(out_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            written += 1

            elapsed = time.time() - t0
            rate = (written - existing) / max(elapsed, 1e-6)
            print(
                f"[{global_idx+1}/{existing+total}] "
                f"ranking={ranking} | {rate:.2f} rec/s | {elapsed/60:.1f} min elapsed"
            )

    elapsed = time.time() - t0
    print(f"\nDone. {written} total records written to {out_path}")
    print(f"Time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
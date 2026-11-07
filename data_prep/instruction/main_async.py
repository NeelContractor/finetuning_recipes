import argparse
import asyncio
import os
import random

from pydantic import BaseModel
from text_albumentations import OpenAIModel, aaugment, arun_augmentation, get_multi_task, save_dataset

from main import (
    PROB_TO_RUN_REPHRASE,
    PROB_TO_RUN_STEP,
    SINGLE_CHUNK_TASKS,
    SINGLE_CHUNK_TASK_PROBABILITIES,
    chunk_text,
    load_texts_from_jsonl,
    selected_single_chunk_tasks,
)

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", help="Path to the input JSONL file")
    parser.add_argument("output_jsonl", help="Path to the output JSONL file")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start processing from this text index in the input JSONL",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of words per text chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Number of overlapping words between consecutive chunks",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.environ.get("TEXT_ALBUMENTATIONS_MODEL", "gpt-5-mini"),
        help="OpenAI-compatible model name for async generation",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "openrouter"],
        default=os.environ.get("LLM_PROVIDER", "openai"),
        help="Which OpenAI-compatible provider client path to use",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Optional OpenAI-compatible base URL override",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI-compatible API key",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop early after generating at least this many rows",
    )
    parser.add_argument(
        "--total-concurrent-calls",
        type=int,
        default=8,
        help="Maximum concurrent async model calls",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for task sampling",
    )
    parser.add_argument(
        "--task-preset",
        choices=["expanded", "legacy"],
        default="expanded",
        help="Use the prior-training single-passage task set without continuation",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["auto", "explicit", "sample"],
        default="sample",
        help=(
            "How text-albumentations selects single-passage tasks. "
            "auto uses the smart switch over the task preset whitelist; "
            "sample uses this repo's task probabilities."
        ),
    )
    parser.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Disable text-albumentations prefilter quality checks",
    )
    parser.add_argument(
        "--postfilter",
        action="store_true",
        help="Enable text-albumentations post-generation quality filtering",
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Disable cross-chunk retrieval generation",
    )
    parser.add_argument(
        "--single-chunk-only",
        action="store_true",
        help=(
            "Flatten all chunks across texts and generate only single-chunk "
            "augmentations"
        ),
    )
    parser.add_argument(
        "--response-format",
        choices=["auto", "json_schema", "json_object"],
        default="auto",
        help="Structured-output mode passed to text-albumentations OpenAIModel",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "disabled"],
        default=os.environ.get("TEXT_ALBUMENTATIONS_REASONING_EFFORT", "low"),
        help="Reasoning effort for compatible endpoints; use disabled for local servers",
    )
    return parser.parse_args()


def resolve_client_config(args):
    if args.provider == "openrouter":
        return {
            "api_key": args.api_key or os.environ.get("OPENROUTER_API_KEY"),
            "base_url": args.base_url or OPENROUTER_BASE_URL,
        }

    return {
        "api_key": args.api_key or os.environ.get("OPENAI_API_KEY"),
        "base_url": (
            args.base_url
            or os.environ.get("OPENAI_BASE_URL")
            or OPENAI_BASE_URL
        ),
    }


def build_async_runtime(args):
    client_kwargs = resolve_client_config(args)
    reasoning_effort = (
        None if args.reasoning_effort == "disabled" else args.reasoning_effort
    )
    return OpenAIModel(
        args.model_name,
        base_url=client_kwargs["base_url"],
        api_key=client_kwargs["api_key"],
        async_mode=True,
        total_concurrent_calls=args.total_concurrent_calls,
        response_format=args.response_format,
        reasoning_effort=reasoning_effort,
    )


class _PassageQuality(BaseModel):
    is_quality: bool


_PREFILTER_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a data quality filter. "
            "Return is_quality=true if the passage contains meaningful natural-language "
            "content (at least ~10 words, not just code, markup, metadata, or gibberish). "
            "Return is_quality=false otherwise."
        ),
    }
]


async def _ais_quality(chunk: str, runtime) -> bool:
    messages = _PREFILTER_MESSAGES + [{"role": "user", "content": chunk}]
    result = await runtime.agenerate_structured(messages, _PassageQuality, temperature=0.0, max_tokens=1024)
    return result.is_quality


async def aaugment_with_prefilter(
    chunk: str,
    task_spec,
    selection_mode: str,
    runtime,
    prefilter: bool,
    postfilter: bool,
) -> list:
    if prefilter and selection_mode == "sample":
        if not await _ais_quality(chunk, runtime):
            return []
    return await aaugment(
        chunk,
        tasks=task_spec,
        selection_mode=selection_mode,
        model=runtime,
        prefilter=prefilter,
        postfilter=postfilter,
        sample_instruction_template=False,
    )


def build_single_chunk_task_spec(task_preset: str, selection_mode: str):
    selected_task_names = [
        task_name
        for task_name, _, _ in selected_single_chunk_tasks(task_preset)
    ]
    if selection_mode == "sample":
        return {
            task_name: SINGLE_CHUNK_TASK_PROBABILITIES[task_name]
            for task_name in selected_task_names
        }
    return selected_task_names


async def generate_examples_for_chunks_async(
    chunks: list[str],
    runtime,
    task_preset: str = "expanded",
    selection_mode: str = "sample",
    prefilter: bool = True,
    postfilter: bool = False,
):
    tasks = []
    task_spec = build_single_chunk_task_spec(task_preset, selection_mode)

    print(
        f"Scheduling async single-passage augmentation with "
        f"selection_mode={selection_mode} task_count={len(task_spec)}"
    )
    for chunk in chunks:
        tasks.append(
            aaugment_with_prefilter(
                chunk,
                task_spec,
                selection_mode,
                runtime,
                prefilter,
                postfilter,
            )
        )

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    dataset = []
    for result in results:
        if isinstance(result, KeyboardInterrupt):
            raise result
        if isinstance(result, Exception):
            print(f"Skipping async augmentation: {result}")
            continue
        dataset.extend(result)
    return dataset


async def generate_and_save_examples_for_chunks_async(
    chunks: list[str],
    runtime,
    output_jsonl: str,
    task_preset: str = "expanded",
    selection_mode: str = "sample",
    prefilter: bool = True,
    postfilter: bool = False,
):
    task_spec = build_single_chunk_task_spec(task_preset, selection_mode)

    print(
        f"Scheduling async single-passage augmentation with "
        f"selection_mode={selection_mode} task_count={len(task_spec)}"
    )

    async def augment_chunk(chunk_idx: int, chunk: str):
        dataset = await aaugment_with_prefilter(
            chunk,
            task_spec,
            selection_mode,
            runtime,
            prefilter,
            postfilter,
        )
        return chunk_idx, dataset

    tasks = [
        asyncio.create_task(augment_chunk(chunk_idx, chunk))
        for chunk_idx, chunk in enumerate(chunks, start=1)
    ]
    if not tasks:
        return 0

    total_rows = 0
    for completed in asyncio.as_completed(tasks):
        try:
            chunk_idx, dataset = await completed
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"Skipping async augmentation: {exc}")
            continue

        total_rows += len(dataset)
        save_dataset(dataset, output_jsonl)
        print(
            f"Saved {len(dataset)} single-chunk rows for chunk "
            f"{chunk_idx}/{len(chunks)}"
        )

    return total_rows


def iter_selected_chunks(
    texts: list[str],
    chunk_size: int,
    chunk_overlap: int,
):
    for text_idx, text in enumerate(texts, start=1):
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            print(f"Skipping text {text_idx}: no valid chunks")
            continue

        selected_chunks = chunks[: max(1, int(len(chunks) // 2))]
        for chunk_idx, chunk in enumerate(selected_chunks, start=1):
            yield text_idx, chunk_idx, len(selected_chunks), chunk


async def generate_and_save_flat_single_chunks_async(
    texts: list[str],
    runtime,
    args,
):
    task_spec = build_single_chunk_task_spec(args.task_preset, args.selection_mode)
    chunk_iter = iter_selected_chunks(texts, args.chunk_size, args.chunk_overlap)
    max_chunk_jobs = max(1, args.total_concurrent_calls)
    pending = set()
    scheduled_chunks = 0
    completed_chunks = 0
    total_examples = 0
    exhausted = False

    async def augment_chunk(text_idx, chunk_idx, chunks_for_text, chunk):
        dataset = await aaugment_with_prefilter(
            chunk,
            task_spec,
            args.selection_mode,
            runtime,
            not args.no_prefilter,
            args.postfilter,
        )
        return text_idx, chunk_idx, chunks_for_text, dataset

    def schedule_until_full():
        nonlocal exhausted
        nonlocal scheduled_chunks
        while not exhausted and len(pending) < max_chunk_jobs:
            try:
                text_idx, chunk_idx, chunks_for_text, chunk = next(chunk_iter)
            except StopIteration:
                exhausted = True
                break
            scheduled_chunks += 1
            pending.add(
                asyncio.create_task(
                    augment_chunk(text_idx, chunk_idx, chunks_for_text, chunk)
                )
            )

    print(
        "Running flattened single-chunk generation with "
        f"max_chunk_jobs={max_chunk_jobs} selection_mode={args.selection_mode} "
        f"task_count={len(task_spec)}"
    )
    schedule_until_full()

    while pending:
        done, pending = await asyncio.wait(
            pending,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for completed in done:
            completed_chunks += 1
            try:
                text_idx, chunk_idx, chunks_for_text, dataset = await completed
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"Skipping async augmentation: {exc}")
                continue

            total_examples += len(dataset)
            save_dataset(dataset, args.output_jsonl)
            print(
                f"Saved {len(dataset)} rows from text {text_idx} "
                f"chunk {chunk_idx}/{chunks_for_text}. "
                f"total_rows={total_examples} completed_chunks={completed_chunks} "
                f"scheduled_chunks={scheduled_chunks}"
            )

        if args.max_rows is not None and total_examples >= args.max_rows:
            exhausted = True
        schedule_until_full()

    print(
        f"Flattened single-chunk generation saved {total_examples} rows "
        f"from {completed_chunks} completed chunks."
    )
    return total_examples


async def generate_cross_chunk_examples_async(
    chunks: list[str],
    runtime,
    retrieval: bool = True,
):
    tasks = []

    if len(chunks) >= 2:
        if retrieval and random.random() < PROB_TO_RUN_STEP:
            print("Scheduling async retrieval")
            tasks.append(arun_augmentation(chunks, get_multi_task("retrieval"), runtime))

        if random.random() < PROB_TO_RUN_REPHRASE:
            left_idx, right_idx = random.sample(range(len(chunks)), 2)
            print("Scheduling async comparison")
            tasks.append(
                arun_augmentation(
                    [chunks[left_idx], chunks[right_idx]],
                    get_multi_task("comparison"),
                    runtime,
                )
            )

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    dataset = []
    for result in results:
        if isinstance(result, KeyboardInterrupt):
            raise result
        if isinstance(result, Exception):
            print(f"Skipping async augmentation: {result}")
            continue
        dataset.extend(result)
    return dataset


async def amain():
    args = parse_args()
    random.seed(args.seed)

    runtime = build_async_runtime(args)

    texts = load_texts_from_jsonl(args.input_jsonl)
    print(f"Loaded {len(texts)} texts from {args.input_jsonl}")
    print(f"provider={args.provider}")
    print(f"model_name={args.model_name}")
    print(f"base_url={resolve_client_config(args)['base_url']}")
    print(f"total_concurrent_calls={args.total_concurrent_calls}")
    print(f"task_preset={args.task_preset}")
    print(f"selection_mode={args.selection_mode}")
    print(f"prefilter={not args.no_prefilter}")
    print(f"postfilter={args.postfilter}")
    print(f"retrieval={not args.no_retrieval}")
    print(f"single_chunk_only={args.single_chunk_only}")
    print(
        "single_chunk_tasks="
        f"{len(selected_single_chunk_tasks(args.task_preset))}/"
        f"{len(SINGLE_CHUNK_TASKS)}"
    )

    total_chunks = 0
    total_examples = 0
    texts = texts[args.start_index:]

    if args.single_chunk_only:
        await generate_and_save_flat_single_chunks_async(texts, runtime, args)
        return

    for text_idx, text in enumerate(texts, start=1):
        print(f"Processing text {text_idx}/{len(texts)}")
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        if not chunks:
            print(f"Skipping text {text_idx}: no valid chunks")
            continue

        total_chunks += len(chunks)
        chunks = chunks[: max(1, int(len(chunks) // 2))]

        row_count = await generate_and_save_examples_for_chunks_async(
            chunks,
            runtime,
            args.output_jsonl,
            args.task_preset,
            args.selection_mode,
            not args.no_prefilter,
            args.postfilter,
        )
        print(
            f"Generated and saved {row_count} single-chunk rows "
            f"for text {text_idx}"
        )

        total_examples += row_count
        print(f"Total rows saved so far: {total_examples}")

        if args.max_rows is not None and total_examples >= args.max_rows:
            print(f"Reached max row limit of {args.max_rows}. Stopping early.")
            break

        cross_chunk_dataset = await generate_cross_chunk_examples_async(
            chunks,
            runtime,
            not args.no_retrieval,
        )
        print(
            f"Generated {len(cross_chunk_dataset)} cross-chunk rows for text "
            f"{text_idx} before saving"
        )

        total_examples += len(cross_chunk_dataset)
        save_dataset(cross_chunk_dataset, args.output_jsonl)
        print(f"Total rows saved so far: {total_examples}")

        if args.max_rows is not None and total_examples >= args.max_rows:
            print(f"Reached max row limit of {args.max_rows}. Stopping early.")
            break

    print(
        f"Processed {len(texts)} texts into {total_chunks} chunks "
        f"and generated {total_examples} examples."
    )


if __name__ == "__main__":
    asyncio.run(amain())

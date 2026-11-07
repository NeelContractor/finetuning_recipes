import argparse
import json
import random
from text_albumentations import (
    get_default_outlines_runtime,
    run_augmentation,
    save_dataset,
)
from text_albumentations.tasks import (
    bullet_augmentation,
    comparison_augmentation,
    qa_pair_augmentation,
    rephrase_augmentation,
    retrieval_augmentation,
    triplet_augmentation,
)


PROB_TO_RUN_STEP = 0.25
PROB_TO_RUN_REPHRASE = 0.1
RUNTIME = None
SINGLE_CHUNK_TASKS = [
    ("bullets", bullet_augmentation, PROB_TO_RUN_STEP),
    ("qa_pairs", qa_pair_augmentation, PROB_TO_RUN_STEP),
    ("rephrase", rephrase_augmentation, PROB_TO_RUN_REPHRASE),
    ("triplets", triplet_augmentation, PROB_TO_RUN_STEP),
]
SINGLE_CHUNK_TASK_PROBABILITIES = {
    task_name: probability
    for task_name, _, probability in SINGLE_CHUNK_TASKS
}
LEGACY_SINGLE_CHUNK_TASK_NAMES = {
    "bullets",
    "qa_pairs",
    "rephrase",
    "triplets",
}


def get_runtime():
    global RUNTIME
    if RUNTIME is None:
        RUNTIME = get_default_outlines_runtime()
    return RUNTIME


def chunk_text(
    text: str,
    chunk_size_words: int = 500,
    overlap_words: int = 100,
) -> list[str]:
    words = text.split()
    if not words:
        return []

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    step = chunk_size_words - overlap_words

    return [
        " ".join(words[idx: idx + chunk_size_words])
        for idx in range(0, len(words), step)
    ]


def load_texts_from_jsonl(path: str) -> list[str]:
    texts = []

    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            row = json.loads(raw)
            text = row.get("text", "")
            if not isinstance(text, str):
                raise ValueError(
                    f"Expected 'text' to be a string on line {line_number}"
                )

            text = text.strip()
            if text:
                texts.append(text)

    return texts


def try_generate(label: str, fn):
    try:
        return fn()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"Skipping {label}: {exc}")
        return []


def selected_single_chunk_tasks(task_preset: str):
    if task_preset == "legacy":
        return [
            task
            for task in SINGLE_CHUNK_TASKS
            if task[0] in LEGACY_SINGLE_CHUNK_TASK_NAMES
        ]
    return SINGLE_CHUNK_TASKS


def generate_examples_for_chunk(chunk: str, task_preset: str = "expanded"):
    dataset = []

    for task_name, augmentation, probability in selected_single_chunk_tasks(
        task_preset
    ):
        if random.random() >= probability:
            continue

        print(f"Generating {task_name}")
        dataset.extend(
            try_generate(
                task_name,
                lambda augmentation=augmentation: run_augmentation(
                    chunk,
                    augmentation,
                    get_runtime(),
                ),
            )
        )

    return dataset


def generate_cross_chunk_examples(chunks: list[str]):
    dataset = []

    if len(chunks) >= 2:
        print("Generating retrieval")
        if random.random() < PROB_TO_RUN_STEP:
            dataset.extend(
                try_generate(
                    "retrieval",
                    lambda: run_augmentation(
                        chunks,
                        retrieval_augmentation,
                        get_runtime(),
                    ),
                )
            )

        if random.random() < PROB_TO_RUN_REPHRASE:
            left_idx, right_idx = random.sample(range(len(chunks)), 2)
            print("Generating comparisons")
            dataset.extend(
                try_generate(
                    "comparison",
                    lambda: run_augmentation(
                        [chunks[left_idx], chunks[right_idx]],
                        comparison_augmentation,
                        get_runtime(),
                    ),
                )
            )

    return dataset


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
        "--max-rows",
        type=int,
        default=None,
        help="Stop early after generating at least this many rows",
    )
    parser.add_argument(
        "--task-preset",
        choices=["expanded", "legacy"],
        default="expanded",
        help="Use the prior-training single-passage task set without continuation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    texts = load_texts_from_jsonl(args.input_jsonl)
    print(f"Loaded {len(texts)} texts from {args.input_jsonl}")

    total_chunks = 0
    total_examples = 0
    texts = texts[args.start_index:]

    for text_idx, text in enumerate(texts, start=1):
        print(f"Processing text {text_idx}/{len(texts)}")
        chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        if not chunks:
            print(f"Skipping text {text_idx}: no valid chunks")
            continue

        total_chunks += len(chunks)
        print(
            f"Generated {len(chunks)} chunks for text {text_idx}. "
            f"Total chunks so far: {total_chunks}"
        )

        print(f"Generating synthetic examples for text {text_idx}")
        rows_generated_for_text = 0

        total_chunks = len(chunks)
        chunks = chunks[: int(total_chunks // 2)]
        for chunk_idx, chunk in enumerate(chunks, start=1):
            print(f"Processing chunk {chunk_idx}/{len(chunks)} for text {text_idx}")
            dataset = []
            try:
                dataset = generate_examples_for_chunk(chunk, args.task_preset)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"Errored out for chunk {chunk_idx}: {exc}\n")
            print(
                f"Generated {len(dataset)} rows for chunk {chunk_idx} "
                f"of text {text_idx} before saving"
            )

            total_examples += len(dataset)
            rows_generated_for_text += len(dataset)
            save_dataset(dataset, args.output_jsonl)
            print(f"Total rows saved so far: {total_examples}")

            if args.max_rows is not None and total_examples >= args.max_rows:
                print(f"Reached max row limit of {args.max_rows}. Stopping early.")
                break

        if args.max_rows is not None and total_examples >= args.max_rows:
            break

        dataset = generate_cross_chunk_examples(chunks)
        print(
            f"Generated {len(dataset)} cross-chunk rows for text {text_idx} "
            f"before saving"
        )

        total_examples += len(dataset)
        rows_generated_for_text += len(dataset)
        save_dataset(dataset, args.output_jsonl)
        print(
            f"Generated {rows_generated_for_text} total rows for text {text_idx}"
        )
        print(f"Total rows saved so far: {total_examples}")

        if args.max_rows is not None and total_examples >= args.max_rows:
            print(f"Reached max row limit of {args.max_rows}. Stopping early.")
            break

    print(
        f"Processed {len(texts)} texts into {total_chunks} chunks "
        f"and generated {total_examples} examples."
    )


if __name__ == "__main__":
    main()

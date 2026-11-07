"""One-word-swap detection benchmark for reward models.

For each reference, compare score(ref, ref) against score(ref, ref-with-one-word
changed). A model "notices" a meaning-changing edit when its score drops by more
than --threshold relative to the exact-match score for the same reference.

Usage:
    python reward_models/benchmark_swap.py \
        --models paperbd/neuraltxt-reward-22M \
        --num_samples 150 --output reward_model_swap.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from augment import (  # noqa: E402
    ANTONYM_MAP,
    SYNONYM_MAP,
    antonym_swap,
    negation_flip,
    number_swap,
    synonym_swap,
)
from reward_model import load_reward_model  # noqa: E402

RANDOM_WORDS = [
    "banana", "telescope", "violin", "glacier", "harbor",
    "lantern", "meadow", "pyramid", "quartz", "rocket",
    "saddle", "thunder", "umbrella", "violet", "wagon",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Reward-model swap benchmark.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["paperbd/neuraltxt-reward-22M"],
        help="Reward-model directories or HF repo ids.",
    )
    parser.add_argument(
        "--dataset",
        default="paperbd/paper_answers_reward",
        help="HF dataset with reference answers.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_samples", type=int, default=150)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reward_model_swap_results.json")
    return parser.parse_args()


def get_references(args):
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    refs = []
    seen = set()
    for record in dataset:
        text = (
            record.get("orig_reference_answer")
            or record.get("reference")
            or record.get("output")
            or record.get("answer")
        )
        if not isinstance(text, str):
            continue
        text = text.strip()
        if len(text.split()) < 8 or text in seen:
            continue
        seen.add(text)
        refs.append(text)
    rng = random.Random(args.seed)
    rng.shuffle(refs)
    return refs[: args.num_samples]


def random_word_swap(text, rng):
    """Replace one word with an unrelated word (not in augment vocab)."""
    words = text.split()
    candidates = [
        i
        for i, word in enumerate(words)
        if word.lower().strip(".,;:!?()[]") not in ANTONYM_MAP
        and word.lower().strip(".,;:!?()[]") not in SYNONYM_MAP
        and word.isalpha()
    ]
    if not candidates:
        return None
    index = rng.choice(candidates)
    replacement = rng.choice(RANDOM_WORDS)
    if words[index][0].isupper():
        replacement = replacement.capitalize()
    words[index] = replacement
    return " ".join(words)


def make_perturbations(reference, seed):
    rng = random.Random(seed)
    return {
        "negation": negation_flip(reference, rng),
        "number": number_swap(reference, rng),
        "antonym": antonym_swap(reference, rng),
        "random_word": random_word_swap(reference, rng),
        "synonym": synonym_swap(reference, rng),
    }


def evaluate(model, references, threshold):
    exact_scores = model.score_batch(references, references)

    per_type = {}
    for perturbation in [
        "negation",
        "number",
        "antonym",
        "random_word",
        "synonym",
    ]:
        refs = []
        candidates = []
        baselines = []
        for index, reference in enumerate(references):
            candidate = make_perturbations(
                reference,
                f"{index}:{perturbation}",
            )[perturbation]
            if not candidate or candidate == reference:
                continue
            refs.append(reference)
            candidates.append(candidate)
            baselines.append(exact_scores[index])

        if not candidates:
            per_type[perturbation] = {
                "n": 0,
                "mean_score": None,
                "noticed_rate": None,
            }
            continue

        scores = model.score_batch(refs, candidates)
        noticed = sum(
            1
            for baseline, score in zip(baselines, scores)
            if baseline - score > threshold
        )
        per_type[perturbation] = {
            "n": len(scores),
            "mean_score": sum(scores) / len(scores),
            "noticed_rate": noticed / len(scores),
        }

    return {
        "n": len(references),
        "threshold": threshold,
        "exact_match": {
            "mean_score": sum(exact_scores) / len(exact_scores),
            "above_0_8_rate": sum(1 for s in exact_scores if s > 0.8)
            / len(exact_scores),
        },
        "perturbations": per_type,
    }


def main():
    args = parse_args()
    references = get_references(args)
    if not references:
        raise SystemExit("No usable references found in the dataset.")
    print(f"References: {len(references)}")

    results = {}
    for model_path in args.models:
        print(f"\nLoading {model_path}...")
        model = load_reward_model(model_path)[0]
        results[model_path] = evaluate(model, references, args.threshold)

    output = {
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(references),
        "results": results,
    }
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print(f"One-Word-Swap Detection (n={len(references)})")
    print("=" * 78)
    header = f"{'Perturbation':<22}"
    for model_path in results:
        header += f"{Path(model_path).name:>26}"
    print(header)
    print("-" * 78)
    for perturbation in [
        "exact_match",
        "negation",
        "number",
        "antonym",
        "random_word",
        "synonym",
    ]:
        line = f"{perturbation:<22}"
        for model_path in results:
            if perturbation == "exact_match":
                data = results[model_path]["exact_match"]
                value = f"{data['mean_score']:.3f}"
                rate = data["above_0_8_rate"]
                line += f"{value:>16} ({rate:>5.0%})"
            else:
                data = results[model_path]["perturbations"][perturbation]
                if data["noticed_rate"] is None:
                    line += f"{'n/a':>26}"
                else:
                    line += (
                        f"{data['mean_score']:>16.3f} "
                        f"{data['noticed_rate']:>6.0%}"
                    )
        print(line)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

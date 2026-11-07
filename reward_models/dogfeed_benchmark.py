"""Dogfeed benchmark: hand-written extreme cases for reward-model sanity checks.

This is intentionally not a statistical benchmark. It checks dynamic range:
an identical response should score high, a completely irrelevant one low, and a
flat/HuggingFace-style model that compresses everything into 0.4-0.7 is a red
flag even when its Spearman correlation looks fine.

Usage:
    python reward_models/dogfeed_benchmark.py \
        --models paperbd/neuraltxt-reward-22M
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reward_model import load_reward_model  # noqa: E402


PASSAGE = (
    "The proposed method reduces inference latency by 42% while improving "
    "accuracy by 3.1 points on the benchmark. The authors attribute the gain "
    "to sparse attention and a smaller reward-model head."
)

CASES = [
    {
        "name": "1:1 identical",
        "reference": PASSAGE,
        "response": PASSAGE,
    },
    {
        "name": "exact copy",
        "reference": PASSAGE,
        "response": PASSAGE,
    },
    {
        "name": "one-word swap",
        "reference": PASSAGE,
        "response": PASSAGE.replace("proposed", "existing"),
    },
    {
        "name": "negation",
        "reference": PASSAGE,
        "response": PASSAGE.replace("reduces", "does not reduce"),
    },
    {
        "name": "wrong number",
        "reference": PASSAGE,
        "response": PASSAGE.replace("42%", "24%"),
    },
    {
        "name": "wrong format",
        "reference": PASSAGE,
        "response": json.dumps({"summary": PASSAGE}),
    },
    {
        "name": "half response",
        "reference": PASSAGE,
        "response": " ".join(PASSAGE.split()[:12]),
    },
    {
        "name": "repeated 2x",
        "reference": PASSAGE,
        "response": f"{PASSAGE} {PASSAGE}",
    },
    {
        "name": "completely irrelevant",
        "reference": PASSAGE,
        "response": "The capital of France is Paris and its population is large.",
    },
    {
        "name": "correct paraphrase",
        "reference": PASSAGE,
        "response": (
            "Sparse attention and a smaller reward head let the method cut "
            "latency by 42% and raise benchmark accuracy by 3.1 points."
        ),
    },
    {
        "name": "good Q&A pair",
        "reference": (
            "Question: What is the main contribution? "
            "Answer: A 22M MiniLM reward model trained with confounds."
        ),
        "response": (
            "Question: What is the main contribution? "
            "Answer: A 22M MiniLM reward model trained with confounds."
        ),
    },
    {
        "name": "verbose bloated",
        "reference": PASSAGE,
        "response": (
            "In summary, it is important to note that the proposed method, "
            "which is notably effective, reduces inference latency by 42% "
            "while also, as shown, improving accuracy by 3.1 points on the "
            "benchmark, and the authors attribute this gain to sparse "
            "attention and a smaller reward-model head, as previously noted."
        ),
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Reward-model dogfeed benchmark.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["paperbd/neuraltxt-reward-22M"],
        help="Reward-model directories or HF repo ids.",
    )
    parser.add_argument("--output", default="reward_model_dogfeed_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}
    for model_path in args.models:
        print(f"\nLoading {model_path}...")
        model = load_reward_model(model_path)[0]
        scores = model.score_batch(
            [case["reference"] for case in CASES],
            [case["response"] for case in CASES],
        )
        results[model_path] = {
            case["name"]: float(score)
            for case, score in zip(CASES, scores)
        }

    output = {"results": results}
    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("Dogfeed (dynamic range)")
    print("=" * 78)
    header = f"{'Case':<24}"
    for model_path in results:
        header += f"{Path(model_path).name:>24}"
    print(header)
    print("-" * 78)
    for case in CASES:
        line = f"{case['name']:<24}"
        for model_path in results:
            line += f"{results[model_path][case['name']]:>24.3f}"
        print(line)

    print("\nDynamic range (identical - irrelevant):")
    for model_path, scores in results.items():
        gap = scores["1:1 identical"] - scores["completely irrelevant"]
        print(f"  {Path(model_path).name:<30} {gap:+.3f}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

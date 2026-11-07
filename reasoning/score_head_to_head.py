"""Score two rollout JSONL files with NeuralTxtReward and compare head-to-head."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning.env import _extract_think_content, score_completions  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two rollout JSONL files using NeuralTxtReward."
    )
    parser.add_argument("--file_a", "-a", required=True, help="Reasoning model rollouts (may have <think> tags)")
    parser.add_argument("--file_b", "-b", required=True, help="SFT model rollouts")
    parser.add_argument("--name_a", default=None, help="Label for model A")
    parser.add_argument("--name_b", default=None, help="Label for model B")
    parser.add_argument("--output_dir", default="reasoning/evals/benchmark")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--a_has_think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Model A responses contain <think>...</think> (reasoning model).",
    )
    parser.add_argument(
        "--b_has_think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Model B responses contain <think>...</think> (default: SFT baseline).",
    )
    return parser.parse_args()


def load_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def extract_answer(completion: str) -> str:
    """Extract just the answer from a reasoning completion (after </think>)."""
    _, response, _ = _extract_think_content(completion)
    return response if response else completion


def main():
    from neuraltxt import NeuralTxtReward

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name_a = args.name_a or Path(args.file_a).stem
    name_b = args.name_b or Path(args.file_b).stem

    print(f"Loading A: {args.file_a}", flush=True)
    rows_a = load_jsonl(args.file_a, args.limit)
    print(f"Loading B: {args.file_b}", flush=True)
    rows_b = load_jsonl(args.file_b, args.limit)

    if args.limit:
        rows_a = rows_a[:args.limit]
        rows_b = rows_b[:args.limit]

    assert len(rows_a) == len(rows_b), f"File length mismatch: {len(rows_a)} vs {len(rows_b)}"
    n = len(rows_a)
    print(f"Comparing {n} examples", flush=True)

    answers_a = [extract_answer(r["response"]) for r in rows_a]
    answers_b = [r["response"] for r in rows_b]
    references = [
        str(r.get("ground_truth", r.get("reference", ""))) for r in rows_a
    ]

    print("Scoring with the training reward (flat length cap)...", flush=True)
    reward_model = NeuralTxtReward(backend="hf")
    batch_a = score_completions(
        [r["response"] for r in rows_a],
        references,
        reward_model=reward_model,
        require_think_format=args.a_has_think,
        include_format_rewards=args.a_has_think,
    )
    batch_b = score_completions(
        [r["response"] for r in rows_b],
        references,
        reward_model=reward_model,
        require_think_format=args.b_has_think,
        include_format_rewards=args.b_has_think,
    )
    # Compare raw semantic quality so think-formatted and SFT outputs are on
    # the same scale; totals are reported alongside for the composite view.
    scores_a = batch_a.components["neuraltxt_reward"].tolist()
    scores_b = batch_b.components["neuraltxt_reward"].tolist()
    totals_a = batch_a.total.tolist()
    totals_b = batch_b.total.tolist()

    wins_a = 0
    wins_b = 0
    ties = 0
    diffs = []
    results = []

    for i, (ref, comp_a, comp_b, ans_a, s_a, s_b, t_a, t_b) in enumerate(
        zip(
            references,
            rows_a,
            rows_b,
            answers_a,
            scores_a,
            scores_b,
            totals_a,
            totals_b,
        )
    ):
        s_a_f = float(s_a)
        s_b_f = float(s_b)
        diff = s_a_f - s_b_f

        if diff > 0.001:
            wins_a += 1
        elif diff < -0.001:
            wins_b += 1
        else:
            ties += 1
        diffs.append(diff)

        results.append({
            "id": comp_a.get("id", i),
            "question": comp_a.get("question", ""),
            "reference": ref,
            f"{name_a}_completion": comp_a["response"],
            f"{name_a}_answer": ans_a,
            f"{name_b}_completion": comp_b["response"],
            f"{name_a}_score": s_a_f,
            f"{name_b}_score": s_b_f,
            f"{name_a}_total_reward": float(t_a),
            f"{name_b}_total_reward": float(t_b),
            "delta": diff,
            "winner": name_a if diff > 0.001 else (name_b if diff < -0.001 else "tie"),
        })

    mean_diff = sum(diffs) / n if diffs else 0.0
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / n) if diffs else 0.0

    summary = {
        "model_a": name_a,
        "model_a_file": args.file_a,
        "model_b": name_b,
        "model_b_file": args.file_b,
        "num_samples": n,
        "model_a_mean_score": sum(scores_a) / n,
        "model_b_mean_score": sum(scores_b) / n,
        "model_a_mean_total_reward": sum(totals_a) / n,
        "model_b_mean_total_reward": sum(totals_b) / n,
        "mean_delta": mean_diff,
        "std_delta": std_diff,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": wins_a / n,
        "win_rate_b": wins_b / n,
        "tie_rate": ties / n,
    }

    results_path = output_dir / "head_to_head_results.jsonl"
    summary_path = output_dir / "head_to_head_summary.json"

    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"\n=== NeuralTxtReward Head-to-Head ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nResults: {results_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()

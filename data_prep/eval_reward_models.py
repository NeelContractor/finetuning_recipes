"""Compare pretrained / lexical reward signals against the LLM-judge score.

Usage:
    python data_prep/eval_reward_models.py \
        --dataset instruction_tuning/evals/reward_model_dataset_clean.jsonl
"""
import argparse
import json

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

DEFAULT_DATASET = "instruction_tuning/evals/reward_model_dataset_clean.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reward-model methods.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help="Device for Skywork/RewardBert (default: cuda > mps > cpu).",
    )
    return parser.parse_args()


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def word_f1(ref, hyp):
    rt = set(ref.lower().split())
    ht = set(hyp.lower().split())
    if not ht:
        return 0.0
    p = len(rt & ht) / len(ht) if ht else 0
    rv = len(rt & ht) / len(rt) if rt else 0
    return 2 * p * rv / (p + rv) if (p + rv) > 0 else 0.0


def main():
    args = parse_args()
    device = pick_device(args.device)
    dtype = torch.float16 if device == "cuda" else torch.float32

    records = []
    with open(args.dataset, encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))

    rng = np.random.RandomState(args.seed)
    n = min(args.n, len(records))
    indices = rng.choice(len(records), n, replace=False)
    sample = [records[i] for i in indices]
    print(f"Device: {device}")
    print(f"Sample size: {n}")

    # Compute all scores
    print("Computing Skywork...")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    skywork_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    model_kwargs = {"torch_dtype": dtype, "num_labels": 1}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    rm = AutoModelForSequenceClassification.from_pretrained(
        skywork_name,
        **model_kwargs,
    )
    if device != "cuda":
        rm = rm.to(device)
    tokenizer = AutoTokenizer.from_pretrained(skywork_name)
    skywork_scores = []
    for record in sample:
        conv = [
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["llm_response"]},
        ]
        formatted = tokenizer.apply_chat_template(conv, tokenize=False)
        if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
            formatted = formatted[len(tokenizer.bos_token):]
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)
        with torch.no_grad():
            skywork_scores.append(rm(**inputs).logits[0][0].item())

    print("Computing Word F1...")
    f1_scores = [
        word_f1(r["reference"], r["llm_response"]) for r in sample
    ]

    print("Computing ROUGE-L...")
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores = [
        scorer.score(r["reference"], r["llm_response"])["rougeL"].fmeasure
        for r in sample
    ]

    print("Computing BERTScore...")
    from bert_score import score as bertscore

    refs = [r["reference"] for r in sample]
    hyps = [r["llm_response"] for r in sample]
    _, _, f1_b = bertscore(hyps, refs, lang="en", verbose=False)
    bert_scores = f1_b.tolist()

    print("Computing RewardBert...")
    from qa_metrics.RewardBert import RewardBert

    rb = RewardBert(device=device)
    rb_scores = []
    for record in sample:
        norm, _ = rb.compute_score(
            record["reference"][:500],
            record["llm_response"][:500],
        )
        rb_scores.append(norm)

    gt = [r["score"] / 5.0 for r in sample]
    response_words = [len(str(r["llm_response"]).split()) for r in sample]

    all_methods = {
        "Skywork-V2 (0.6B)": skywork_scores,
        "Word F1": f1_scores,
        "ROUGE-L": rouge_scores,
        "BERTScore": bert_scores,
        "RewardBert": rb_scores,
    }

    print("\n" + "=" * 80)
    print(f"CLEAN DATASET RESULTS (no think tags, n={n})")
    print("=" * 80)
    print(
        f"{'Method':<22} {'Spearman r':>12} {'Kendall tau':>12} "
        f"{'Min':>10} {'Max':>10} {'Mean':>8}"
    )
    print("-" * 80)
    print(
        f"{'Judge':<22} {'-':>12} {'-':>12} "
        f"{min(gt):>10.3f} {max(gt):>10.3f} {np.mean(gt):>8.3f}"
    )
    for name, scores in all_methods.items():
        sr, _ = spearmanr(gt, scores)
        kt, _ = kendalltau(gt, scores)
        values = np.array(scores)
        print(
            f"{name:<22} {sr:>12.4f} {kt:>12.4f} "
            f"{values.min():>10.3f} {values.max():>10.3f} "
            f"{values.mean():>8.3f}"
        )

    # Score buckets
    print("\n" + "=" * 80)
    print("BY SCORE BUCKET (Spearman r)")
    print("=" * 80)
    print(f"{'Method':<22}", end="")
    buckets = [
        ("Low (<=0.4)", lambda x: x <= 0.4),
        ("Mid (0.4-0.8)", lambda x: 0.4 < x < 0.8),
        ("High (>=0.8)", lambda x: x >= 0.8),
    ]
    for bucket_name, fn in buckets:
        count = sum(1 for value in gt if fn(value))
        print(f"  {bucket_name} (n={count})", end="")
    print()
    for name, scores in all_methods.items():
        print(f"{name:<22}", end="")
        for _, fn in buckets:
            idxs = [i for i, value in enumerate(gt) if fn(value)]
            if len(idxs) >= 3:
                sr, _ = spearmanr(
                    [gt[i] for i in idxs],
                    [scores[i] for i in idxs],
                )
                print(f"  r={sr:.3f}".ljust(20), end="")
            else:
                print("  n/a".ljust(20), end="")
        print()

    # Length bias (Spearman r vs response word count)
    print("\n" + "=" * 80)
    print("LENGTH BIAS (Spearman r vs response word count)")
    print("=" * 80)
    judge_len, _ = spearmanr(response_words, gt)
    print(f"{'Judge (baseline)':<22} {judge_len:>12.4f}")
    for name, scores in all_methods.items():
        sr, _ = spearmanr(response_words, scores)
        print(f"{name:<22} {sr:>12.4f}")


if __name__ == "__main__":
    main()

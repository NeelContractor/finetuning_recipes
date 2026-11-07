"""Benchmark a reward model across the paper, confound, and answer-equivalence sets.

Usage:
    python data_prep/benchmark_reward.py \
        --model models/reward_model_final/reward_model_finetuned_v7-1 \
        --eval-dir instruction_tuning/evals
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "reward_models"))

from reward_model import load_reward_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a reward model.")
    parser.add_argument(
        "--model",
        default="models/reward_model_final/reward_model_finetuned_v2",
        help="Local reward-model directory or HF repo id.",
    )
    parser.add_argument(
        "--eval-dir",
        default="instruction_tuning/evals",
        help="Directory containing the 100-point eval JSONL files.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for RewardBert (default: cuda > mps > cpu).",
    )
    parser.add_argument("--limit", type=int, default=None)
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
    p = len(rt & ht) / len(ht)
    rv = len(rt & ht) / len(rt)
    return 2 * p * rv / (p + rv) if (p + rv) > 0 else 0.0


def load_jsonl(path, limit=None):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def score_ours(model, refs, resps):
    return model.score_batch(refs, resps)


def main():
    args = parse_args()
    device = pick_device(args.device)
    eval_dir = Path(args.eval_dir)

    print(f"Loading reward model: {args.model}")
    our = load_reward_model(args.model)[0]

    from qa_metrics.RewardBert import RewardBert

    rb = RewardBert(device=device)

    # ===== 1. 100-point eval =====
    results_path = eval_dir / "smollm_135M_neuraltxt_sft_v2_mlx_results.jsonl"
    judged_path = eval_dir / "smollm_135M_neuraltxt_sft_v2_mlx_results_judged.jsonl"
    if results_path.exists() and judged_path.exists():
        print("=" * 60)
        print("1. 100-POINT EVAL (sft_v2 responses)")
        print("-" * 60)

        records = load_jsonl(results_path, args.limit)
        judge_records = load_jsonl(judged_path, args.limit)
        refs = [r["ground_truth"] for r in records]
        resps = [r["response"] for r in records]
        gt = np.array([sum(r["scores"].values()) / 20.0 for r in judge_records])

        our_scores = score_ours(our, refs, resps)
        rb_scores = [
            rb.compute_score(refs[i][:500], resps[i][:500])[0]
            for i in range(len(refs))
        ]
        f1_scores = [word_f1(refs[i], resps[i]) for i in range(len(refs))]

        for name, scores in [
            ("Ours", our_scores),
            ("RewardBert", rb_scores),
            ("Word F1", f1_scores),
        ]:
            sr, _ = spearmanr(gt, scores)
            print(f"  {name:<12} Spearman r={sr:.4f}  Mean={np.mean(scores):.4f}")
    else:
        print(f"SKIP 100-point eval: missing {results_path} or {judged_path}")

    # ===== 2. Validation confounds =====
    from datasets import load_dataset

    print(f"\n{'=' * 60}")
    print("2. VALIDATION CONFOUNDS")
    print("-" * 60)

    val = load_dataset("paperbd/papers_answer_equivalance", split="test")
    confounds = [r for r in val if r["orig_score"] == 3.0]
    refs_c = [str(r["orig_reference_answer"]) for r in confounds]
    bads_c = [str(r["orig_response"]) for r in confounds]

    our_c = score_ours(our, refs_c, bads_c)
    rb_c = [
        rb.compute_score(refs_c[i][:500], bads_c[i][:500])[0]
        for i in range(len(refs_c))
    ]
    f1_c = [word_f1(refs_c[i], bads_c[i]) for i in range(len(refs_c))]

    for name, scores in [
        ("Ours", our_c),
        ("RewardBert", rb_c),
        ("Word F1", f1_c),
    ]:
        s = np.array(scores)
        fooled = (s > 0.7).sum()
        caught = (s < 0.4).sum()
        print(
            f"  {name:<12} Mean={s.mean():.4f}  "
            f"Fooled(>0.7)={fooled}/{len(s)} ({100 * fooled / len(s):.0f}%)  "
            f"Caught(<0.4)={caught}/{len(s)} ({100 * caught / len(s):.0f}%)"
        )

    # ===== 3. Answer equivalence =====
    print(f"\n{'=' * 60}")
    print("3. ANSWER EQUIVALENCE")
    print("-" * 60)

    ae = load_dataset("kortukov/answer-equivalence-dataset", split="dev")
    ae_refs = ae["reference"]
    ae_cands = ae["candidate"]
    ae_labels = [1 if label == "equivalent" else 0 for label in ae["label"]]

    our_ae = score_ours(our, ae_refs, ae_cands)
    rb_ae = [
        rb.compute_score(ae_refs[i][:500], ae_cands[i][:500])[0]
        for i in range(len(ae_refs))
    ]
    f1_ae = [word_f1(ae_refs[i], ae_cands[i]) for i in range(len(ae_refs))]

    for name, scores in [
        ("Ours", our_ae),
        ("RewardBert", rb_ae),
        ("Word F1", f1_ae),
    ]:
        auc = roc_auc_score(ae_labels, scores)
        s = np.array(scores)
        eq = np.array(ae_labels) == 1
        print(
            f"  {name:<12} ROC-AUC={auc:.4f}  "
            f"Equiv mean={s[eq].mean():.4f}  Not mean={s[~eq].mean():.4f}"
        )

    # ===== 4. Val set from our dataset =====
    print(f"\n{'=' * 60}")
    print("4. VALIDATION (paperbd/papers_answer_equivalance test split)")
    print("-" * 60)

    vrefs = val["orig_reference_answer"]
    vresps = val["orig_response"]
    vlabels = [s / 5.0 for s in val["orig_score"]]

    our_val = score_ours(our, vrefs, vresps)
    sr_v, _ = spearmanr(vlabels, our_val)
    print(
        f"  Spearman r={sr_v:.4f}  Mean pred={np.mean(our_val):.4f}  "
        f"Mean label={np.mean(vlabels):.4f}"
    )


if __name__ == "__main__":
    main()

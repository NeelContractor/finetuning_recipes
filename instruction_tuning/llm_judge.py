import argparse
import asyncio
import json
import os
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import outlines

SEMAPHORE = 20

JUDGE_PROMPT = """\
You are an expert evaluator for AI assistant responses on research and academic questions.

Given a question, the model's response, and a ground truth reference answer, score the response on 4 dimensions from 1 to 5.

Scoring guide:
  1 = Very poor   2 = Poor   3 = Acceptable   4 = Good   5 = Excellent

Dimensions:
- faithfulness:        Does the response contain only factually correct claims? Penalise hallucinations.
- answer_correctness:  How closely does the response match the ground truth semantically?
- relevance:           Does the response directly address what was asked, without padding or going off-topic?
- completeness:        Does the response cover the key points from the ground truth without omitting important details?

---
QUESTION:
{question}

MODEL RESPONSE:
{response}

GROUND TRUTH:
{ground_truth}
---
"""


class JudgeScore(BaseModel):
    # reasoning: str = Field(..., description="Brief explanation of the scores.")
    faithfulness: int = Field(..., ge=1, le=5)
    answer_correctness: int = Field(..., ge=1, le=5)
    relevance: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)


parser = argparse.ArgumentParser(description="Async LLM judge for eval JSONL files.")
parser.add_argument("--model", "-m", type=str, default="nvidia/nemotron-3-super-120b-a12b:free")
parser.add_argument("--input_file", "-i", type=str, required=True)
parser.add_argument("--output_file", "-o", type=str, default=None)
parser.add_argument("--api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY"))
parser.add_argument("--limit", "-n", type=int, default=None)
args = parser.parse_args()

if not args.api_key:
    raise ValueError("Set OPENROUTER_API_KEY or pass --api_key")

if args.output_file is None:
    args.output_file = args.input_file.replace(".jsonl", "_judged.jsonl")

client = AsyncOpenAI(
    api_key=args.api_key,
    base_url="https://openrouter.ai/api/v1",
)
model = outlines.from_openai(client, args.model)


async def judge_one(sem: asyncio.Semaphore, idx: int, total: int, record: dict) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=record["question"],
        response=record["response"][:5000],
        ground_truth=record["ground_truth"][:5000],
    )
    async with sem:
        score = None
        for attempt in range(3):
            try:
                raw = await asyncio.wait_for(
                    model(
                        prompt,
                        JudgeScore,
                        max_tokens=10000,
                        extra_body={"reasoning_effort": "low"},
                    ),
                    timeout=300,
                )
                score = (
                    JudgeScore.model_validate_json(raw) if isinstance(raw, str) else raw
                )
                break
            except Exception as e:
                if attempt == 2:
                    score = JudgeScore(
                        faithfulness=1,
                        answer_correctness=1,
                        relevance=1,
                        completeness=1,
                    )
                else:
                    await asyncio.sleep(1)
        if score is None:
            score = JudgeScore(
                faithfulness=1, answer_correctness=1, relevance=1, completeness=1
            )

    result = {
        **record,
        "scores": {
            "faithfulness": score.faithfulness,
            "answer_correctness": score.answer_correctness,
            "relevance": score.relevance,
            "completeness": score.completeness,
        },
        # "reasoning": score.reasoning,
    }
    avg = sum(result["scores"].values()) / 4
    print(
        f"[{idx+1}/{total}] "
        f"F={score.faithfulness} AC={score.answer_correctness} "
        f"R={score.relevance} C={score.completeness} "
        f"avg={avg:.2f} | {record['response'][:60].replace(chr(10), ' ')}..."
    )
    return result


async def main():
    with open(args.input_file) as f:
        records = [json.loads(line) for line in f]

    if args.limit:
        records = records[: args.limit]

    sem = asyncio.Semaphore(SEMAPHORE)

    # Check for existing progress
    done_ids = set()
    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except json.JSONDecodeError:
                    pass  # skip partial writes from crashes

    pending = [r for r in records if r["id"] not in done_ids]
    tasks = [judge_one(sem, i, len(pending), r) for i, r in enumerate(pending)]

    with open(args.output_file, "a") as out:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            out.write(json.dumps(result) + "\n")
            out.flush()

    # Read all results for summary
    all_results = []
    with open(args.output_file) as f:
        for line in f:
            all_results.append(json.loads(line))
    all_results.sort(key=lambda r: r["id"])

    keys = ["faithfulness", "answer_correctness", "relevance", "completeness"]
    print("\n--- summary ---")
    for k in keys:
        vals = [r["scores"][k] for r in all_results]
        print(
            f"  {k:22s}  avg={sum(vals)/len(vals):.2f}  min={min(vals)}  max={max(vals)}"
        )
    overall = [sum(r["scores"].values()) / 4 for r in all_results]
    print(f"  {'overall':22s}  avg={sum(overall)/len(overall):.2f}")
    print(f"\nSaved to {args.output_file} ({len(all_results)} records)")


asyncio.run(main())
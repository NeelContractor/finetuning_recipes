import argparse
import asyncio
import json
import os
from datasets import load_dataset
from openai import AsyncOpenAI

parser = argparse.ArgumentParser(description="API-based batched eval on paper_instructions.")
parser.add_argument("--model", "-m", type=str, required=True)
parser.add_argument("--num_samples", "-n", type=int, default=100)
parser.add_argument("--batch_size", "-bs", type=int, default=10)
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--output_file", "-o", type=str, default=None)
parser.add_argument("--api_key", type=str, default=os.environ.get("OPENROUTER_API_KEY"))
args = parser.parse_args()

if not args.api_key:
    raise ValueError("Set OPENROUTER_API_KEY or pass --api_key")

client = AsyncOpenAI(api_key=args.api_key, base_url="https://openrouter.ai/api/v1")

dataset = load_dataset("paperbd/paper_instructions_300K-v1", split="test")
dataset = dataset.shuffle(seed=args.seed).select(range(args.num_samples))

os.makedirs("instruction_tuning/evals", exist_ok=True)

if args.output_file is None:
    safe_name = args.model.replace("/", "_").replace(":", "_")
    output_path = f"instruction_tuning/evals/{safe_name}_api_results.jsonl"
else:
    output_path = f"instruction_tuning/evals/{args.output_file}"

examples = list(dataset)
total = len(examples)
sem = asyncio.Semaphore(args.batch_size)

async def generate_one(idx: int, ex: dict) -> dict:
    instruction = ex["instruction"]
    inp = ex.get("input", "")
    question = instruction if not inp else f"{instruction}\n\n{inp}"
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": question}],
                    max_tokens=1024,
                    temperature=0,
                )
                response_text = resp.choices[0].message.content or ""
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"[{idx+1}/{total}] retry {attempt+1}: {e}")
                await asyncio.sleep(2)

    record = {
        "id": idx,
        "question": question,
        "response": response_text,
        "ground_truth": ex["output"],
    }
    print(f"[{idx+1}/{total}] {response_text[:80].replace(chr(10), ' ')}...")
    return record

async def main():
    tasks = [generate_one(i, ex) for i, ex in enumerate(examples)]
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda r: r["id"])

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(results)} results to {output_path}")

asyncio.run(main())
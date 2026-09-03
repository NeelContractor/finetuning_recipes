import argparse
import json
import random
import re
from pathlib import Path

ABSTRACT_RE = re.compile(
    r"Suppose that you have an abstract for a scientific paper:\s*(.*?)\s*"
    r"And you have already written the first three sentences of the full article:\s*"
    r"(.*?)\s*Please generate the next two sentences of the article",
    re.DOTALL,
)

def normalize_text(text: str) -> str:
    return " ".join(text.split())

def extract_passage(row: dict) -> str:
    match = ABSTRACT_RE.search(row.get("input", ""))
    if not match:
        return ""

    abstract = normalize_text(match.group(1))
    article_start = normalize_text(match.group(2))
    continuation = normalize_text(row.get("output", ""))
    return "\n\n".join(
        part for part in [abstract, article_start, continuation] if part
    )

def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                yield json.loads(line)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--prepend-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--max-archive-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    reservoir = []
    seen = 0
    for row in iter_jsonl(args.archive_jsonl):
        passage = extract_passage(row)
        if not passage:
            continue
        seen += 1
        item = {"text": passage}
        if len(reservoir) < args.max_archive_rows:
            reservoir.append(item)
            continue
        idx = random.randrange(seen)
        if idx < args.max_archive_rows:
            reservoir[idx] = item

    rows_written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as output_file:
        for prepend_path in args.prepend_jsonl:
            for row in iter_jsonl(prepend_path):
                text = normalize_text(row.get("text", ""))
                if text:
                    output_file.write(json.dumps({"text": text}) + "\n")
                    rows_written += 1
        
        for row in reservoir:
            output_file.write(json.dumps(row) + "\n")
            rows_written += 1

    print(
        f"archive_seen={seen} sampled_archive_rows={len(reservoir)} "
        f"rows_written={rows_written} output_jsonl={args.output_jsonl}"
    )

if __name__ == "__main__":
    main()
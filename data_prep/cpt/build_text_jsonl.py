import argparse
import json
import re
from pathlib import Path


def process_text(text: str) -> str:
    ref_matches = list(re.finditer(r"(?i)\breferences\b", text))
    if not ref_matches:
        return text

    last_ref = ref_matches[-1]
    ref_start = last_ref.start()
    total_len = len(text)

    if ref_start > 0.7 * total_len:
        thrown_out_len = total_len - ref_start
        if thrown_out_len <= 0.3 * total_len:
            return text[:ref_start]
    else:
        app_match = re.search(r"(?i)\bappendix\b", text[ref_start:])
        if app_match:
            app_start = ref_start + app_match.start()
            thrown_out_len = app_start - ref_start
            if thrown_out_len <= 0.3 * total_len:
                return text[:ref_start] + text[app_start:]

    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    txt_files = sorted(args.input_dir.rglob("*.txt"))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with args.output_jsonl.open("w", encoding="utf-8") as output_file:
        for txt_file in txt_files:
            text = process_text(txt_file.read_text(encoding="utf-8")).strip()
            if not text:
                continue
            output_file.write(json.dumps({"text": text}) + "\n")
            rows_written += 1

    print(f"rows_written={rows_written} output_jsonl={args.output_jsonl}")


if __name__ == "__main__":
    main()
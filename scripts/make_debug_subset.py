import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/VL-Calibration-12K/train.jsonl")
    parser.add_argument("--output", default="data/debug/train_32.jsonl")
    parser.add_argument("--n", type=int, default=32)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.n:
                break

    print(f"Wrote {count} examples to {output_path}")


if __name__ == "__main__":
    main()

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data/debug/train_32.jsonl")
    args = parser.parse_args()

    path = Path(args.path)
    key_counter = Counter()
    n = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key_counter.update(row.keys())
            n += 1

    print("num rows:", n)
    print("\nkeys and counts:")
    for k, v in key_counter.most_common():
        print(f"{k}: {v}/{n}")

    needed = [
        "problem",
        "answer",
        "images",
        "response",
        "ground_truth",
        "vision_entropy",
        "vision_kl",
        "vision_token_count",
        "vision_kl_token_count",
    ]

    print("\nfields checked:")
    for k in needed:
        print(f"{k}: {key_counter[k]}/{n}")


if __name__ == "__main__":
    main()

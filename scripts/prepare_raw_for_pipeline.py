import argparse
import json
from pathlib import Path


def convert_row(row):
    """
    Convert raw ModelScope VL-Calibration-12K row into a project-friendly format.

    Raw fields:
      - problem
      - answer
      - images

    Later GPU/model-generation stages should fill:
      - response
      - vision_entropy
      - vision_kl
      - vision_token_count
      - vision_kl_token_count
    """
    return {
        "problem": row["problem"],
        "answer": row["answer"],
        "ground_truth": row["answer"],
        "images": row["images"],

        # Filled later by generation / visual-certainty computation.
        "response": None,
        "vision_entropy": None,
        "vision_kl": None,
        "vision_token_count": None,
        "vision_kl_token_count": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/debug/train_32.jsonl")
    parser.add_argument("--output", default="data/processed/train_32_pipeline.jsonl")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            raw = json.loads(line)
            converted = convert_row(raw)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} converted rows to {output_path}")


if __name__ == "__main__":
    main()

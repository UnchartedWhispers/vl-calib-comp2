import argparse
import json
from pathlib import Path

import pandas as pd


def make_json_serializable(x):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, list):
        return [make_json_serializable(v) for v in x]
    if isinstance(x, tuple):
        return [make_json_serializable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): make_json_serializable(v) for k, v in x.items()}
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/raw/VL-Calibration-12K/train.jsonl")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)

    with out_path.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            row = make_json_serializable(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

"""Print a model's vocab size by reading its config only (no weight download).

Usage:
    python -m scripts.check_vocab --model Qwen/Qwen3-VL-4B-Instruct

Use the printed number for gen_synthetic.py --vocab so the synthetic logits
match the model you dump real logits from.
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    args = ap.parse_args()
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    # VLMs nest the text config; check both.
    vocab = getattr(cfg, "vocab_size", None)
    if vocab is None and hasattr(cfg, "text_config"):
        vocab = getattr(cfg.text_config, "vocab_size", None)
    print(f"{args.model} vocab_size = {vocab}")


if __name__ == "__main__":
    main()

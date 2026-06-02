"""Tier 1: synthetic logit generator (the primary roofline benchmark input).

A roofline study depends on tensor SHAPE and DTYPE, not semantic content. We
generate [T, V] logit pairs with a realistic vocab size and realistic peakiness
(logits are NOT uniform: a few dominant tokens, so softmax/entropy/KL are
non-degenerate). Fully reproducible with a fixed seed; sweep T for the roofline.

The perturbed logits are the original plus controlled noise, mimicking the
distribution shift induced by perturbing the input image.
"""
from __future__ import annotations
import argparse, os
import torch

# Qwen2-VL vocab; override with --vocab for other models.
DEFAULT_VOCAB = 151936


def make_pair(T: int, V: int = DEFAULT_VOCAB, *, peak: float = 8.0,
              n_modes: int = 5, pert_scale: float = 1.5,
              dtype=torch.float16, seed: int = 0, device="cpu"):
    """Return (logits_orig, logits_pert), each [T, V] in `dtype`.

    peak       : magnitude of the dominant-token bumps (controls peakiness)
    n_modes    : how many tokens per row get a bump (a realistic few-mode dist)
    pert_scale : std of the perturbation added to make logits_pert
    """
    g = torch.Generator(device=device).manual_seed(seed)
    base = torch.randn(T, V, generator=g, device=device)            # diffuse background
    idx = torch.randint(0, V, (T, n_modes), generator=g, device=device)
    bumps = peak * torch.rand(T, n_modes, generator=g, device=device)
    logits_orig = base.clone()
    logits_orig.scatter_add_(1, idx, bumps)                          # inject dominant tokens
    noise = pert_scale * torch.randn(T, V, generator=g, device=device)
    logits_pert = logits_orig + noise                                # perturbed-image shift
    return logits_orig.to(dtype), logits_pert.to(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096],
                    help="token-count sweep for the roofline")
    ap.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synth")
    args = ap.parse_args()
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    os.makedirs(args.out, exist_ok=True)
    for T in args.T:
        lo, lp = make_pair(T, args.vocab, dtype=dt, seed=args.seed)
        path = os.path.join(args.out, f"synth_T{T}_V{args.vocab}_{args.dtype}.pt")
        torch.save({"logits_orig": lo, "logits_pert": lp,
                    "T": T, "V": args.vocab, "dtype": args.dtype, "seed": args.seed}, path)
        print(f"wrote {path}  ({lo.shape} {lo.dtype})")


if __name__ == "__main__":
    main()

"""Benchmark driver: time and validate the three visual-certainty arms.

Arms (see src/ops.py and src/triton_kernel.py):
  * eager   -- naive multi-pass PyTorch (DRAM-traffic baseline)
  * compile -- torch.compile(vc_eager): Inductor fuses the passes
  * triton  -- hand-fused single-stream kernel

What it does, per the README B4 spec:
  1. load data/synth/synth_T{T}_V{V}_{dtype}.pt
  2. warm up, then time the chosen arm with CUDA events
  3. verify all three arms agree numerically (entropy and KL)
  4. emit roofline / speedup data into results/

Usage:
  python -m scripts.bench --arm eager  --T 4096
  python -m scripts.bench --arm triton --T 4096
  python -m scripts.bench --all --T 256 512 1024 2048 4096   # full sweep + csv
  python -m scripts.bench --all --T 4096 --save-roofline     # + roofline.json for plotting

Designed to sit under `nsys profile` / `ncu` (single --arm, single --T).
"""
from __future__ import annotations
import argparse
import csv
import glob
import os

import torch

from src.ops import vc_eager, vc_compile, reduce_sample

try:
    from src.triton_kernel import vc_triton, HAVE_TRITON
except Exception:  # pragma: no cover
    HAVE_TRITON = False
    vc_triton = None


ARMS = {
    "eager": vc_eager,
    "compile": vc_compile,
    "triton": vc_triton,
}

_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}


def find_pt(synth_dir: str, T: int, vocab: int | None, dtype: str | None) -> str:
    """Resolve the synthetic .pt for a given T (vocab/dtype optional disambiguators)."""
    v = vocab if vocab is not None else "*"
    d = dtype if dtype is not None else "*"
    pattern = os.path.join(synth_dir, f"synth_T{T}_V{v}_{d}.pt")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No synthetic file matching {pattern}. "
            f"Generate it first: python -m scripts.gen_synthetic --T {T} ..."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous: {matches}. Pin it with --vocab and --dtype."
        )
    return matches[0]


def load_pair(path: str, device: str):
    d = torch.load(path, map_location="cpu")
    lo = d["logits_orig"].to(device)
    lp = d["logits_pert"].to(device)
    return lo, lp, d


def time_arm(fn, lo, lp, *, warmup: int, iters: int, device: str):
    """Return mean ms/iter. CUDA-event timing on GPU, perf_counter on CPU."""
    for _ in range(warmup):
        fn(lo, lp)

    if device == "cuda":
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn(lo, lp)
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]  # ms
    else:
        import time
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(lo, lp)
            times.append((time.perf_counter() - t0) * 1e3)

    times.sort()
    return times[len(times) // 2]  # median ms


def bytes_traffic(T: int, V: int, dtype: str, fused: bool) -> int:
    """Lower-bound DRAM traffic (bytes) for the arm.

    Both logit tensors are [T, V]. The fused kernel reads each element a small
    constant number of times (orig + pert, our 3 passes stay in L2/regs => count
    as one full read of each tensor). The eager arm materializes intermediates
    (logp, p, logq) => ~5 passes. This is the roofline x-axis denominator.
    """
    elem = _DTYPE_BYTES[dtype]
    one_tensor = T * V * elem
    if fused:
        return 2 * one_tensor            # read orig + pert once each
    return 5 * one_tensor                # ~5 memory-bound passes (see ops.py)


def flops(T: int, V: int) -> int:
    """Approximate useful FLOPs for the visual-certainty reduction (per arm-invariant).

    The arithmetic the kernel must do, regardless of how passes are fused
    (counting exp/log as ~1 flop each, which is the convention for roofline
    arithmetic-intensity estimates):

      log_softmax(orig): max + sub + exp + sum + log + sub   ~ 6 V
      softmax prob = exp(...)                                ~ 1 V
      entropy  = sum(p * logp)        : mul + add            ~ 2 V
      log_softmax(pert) for logq                             ~ 6 V
      kl = sum(p * (logp - logq))     : sub + mul + add      ~ 3 V

    => ~18 FLOP per (token, vocab) element. The constant is approximate and
    identical across arms, so it cancels in cross-arm comparisons but sets the
    roofline x-axis (FLOP / byte) honestly.
    """
    return 18 * T * V


def validate(lo, lp, atol_ent=2e-3, atol_kl=2e-3):
    """Check all available arms agree with the eager reference."""
    e_ref, k_ref = vc_eager(lo, lp)
    report = {}
    for name, fn in ARMS.items():
        if fn is None:
            report[name] = "skipped (unavailable)"
            continue
        if name == "triton" and (not HAVE_TRITON or not lo.is_cuda):
            report[name] = "skipped (no triton/GPU)"
            continue
        e, k = fn(lo, lp)
        de = (e.float() - e_ref).abs().max().item()
        dk = (k.float() - k_ref).abs().max().item()
        ok = de <= atol_ent and dk <= atol_kl
        report[name] = f"{'OK ' if ok else 'FAIL'} max|dH|={de:.2e} max|dKL|={dk:.2e}"
    return report


def run(args):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    os.makedirs(args.results, exist_ok=True)

    arms = list(ARMS.keys()) if args.all else [args.arm]
    rows = []

    for T in args.T:
        path = find_pt(args.synth, T, args.vocab, args.dtype)
        lo, lp, meta = load_pair(path, device)
        V, dt = meta["V"], meta["dtype"]

        if args.validate or args.all:
            rep = validate(lo, lp)
            print(f"[validate T={T}] " + " | ".join(f"{k}: {v}" for k, v in rep.items()))

        for arm in arms:
            fn = ARMS[arm]
            if fn is None or (arm == "triton" and (not HAVE_TRITON or device != "cuda")):
                print(f"[skip] arm={arm} unavailable on this device")
                continue

            ms = time_arm(fn, lo, lp, warmup=args.warmup, iters=args.iters, device=device)
            fused = arm in ("compile", "triton")
            byts = bytes_traffic(T, V, dt, fused=fused)
            flop = flops(T, V)
            gbps = byts / (ms * 1e-3) / 1e9          # effective DRAM GB/s
            gflops = flop / (ms * 1e-3) / 1e9        # achieved GFLOP/s
            intensity = flop / byts                  # FLOP / byte (roofline x-axis)
            e, k = fn(lo, lp)
            ve, vk = reduce_sample(e, k)

            print(f"[bench] arm={arm:7s} T={T:5d} V={V} dt={dt} "
                  f"{ms:8.3f} ms  {gbps:8.1f} GB/s  {gflops:8.1f} GFLOP/s  "
                  f"AI={intensity:5.2f}  "
                  f"vision_entropy={ve:.4f} vision_kl={vk:.4f}")
            rows.append({
                "arm": arm, "T": T, "V": V, "dtype": dt,
                "ms": round(ms, 4), "eff_GBps": round(gbps, 2),
                "GFLOPs": round(gflops, 2), "arith_intensity": round(intensity, 4),
                "bytes": byts, "flops": flop,
                "vision_entropy": round(ve, 6), "vision_kl": round(vk, 6),
            })

    if rows:
        out = os.path.join(args.results, "bench.csv")
        write_header = not os.path.exists(out) or args.overwrite
        mode = "w" if write_header else "a"
        with open(out, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {out}")

        # speedup summary vs eager, per T
        by_T = {}
        for r in rows:
            by_T.setdefault(r["T"], {})[r["arm"]] = r["ms"]
        for T, d in sorted(by_T.items()):
            if "eager" in d:
                base = d["eager"]
                sp = {a: f"{base / m:.2f}x" for a, m in d.items() if a != "eager"}
                if sp:
                    print(f"  speedup vs eager @T={T}: " + ", ".join(f"{a} {s}" for a, s in sp.items()))

        if args.save_roofline:
            import json
            roof = {
                "hardware": {
                    "name": args.gpu_name,
                    "peak_dram_GBps": args.peak_GBps,
                    "peak_compute_GFLOPs": args.peak_GFLOPs,
                    "ridge_point_FLOP_per_byte": round(
                        args.peak_GFLOPs / args.peak_GBps, 4),
                },
                "points": [
                    {
                        "arm": r["arm"], "T": r["T"], "V": r["V"], "dtype": r["dtype"],
                        "arith_intensity": r["arith_intensity"],   # x-axis: FLOP/byte
                        "GFLOPs": r["GFLOPs"],                      # y-axis: achieved
                        "eff_GBps": r["eff_GBps"],
                        "ms": r["ms"],
                    }
                    for r in rows
                ],
            }
            rp = os.path.join(args.results, "roofline.json")
            with open(rp, "w") as f:
                json.dump(roof, f, indent=2)
            print(f"wrote roofline points -> {rp}  "
                  f"(ridge @ {roof['hardware']['ridge_point_FLOP_per_byte']} FLOP/byte)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS.keys()), default="eager")
    ap.add_argument("--all", action="store_true", help="run every arm (sweep + validate)")
    ap.add_argument("--T", type=int, nargs="+", default=[4096])
    ap.add_argument("--vocab", type=int, default=None, help="disambiguate .pt by vocab")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None,
                    help="disambiguate .pt by dtype")
    ap.add_argument("--synth", default="data/synth")
    ap.add_argument("--results", default="results")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--validate", action="store_true", help="check arms agree before timing")
    ap.add_argument("--overwrite", action="store_true", help="overwrite bench.csv instead of append")
    ap.add_argument("--cpu", action="store_true", help="force CPU (eager/compile only)")
    ap.add_argument("--save-roofline", action="store_true",
                    help="dump results/roofline.json: (arith_intensity, GFLOPs) points + ceilings")
    ap.add_argument("--peak-GBps", type=float, default=936.0,
                    help="peak DRAM bandwidth for the roofline (default: RTX 3090 ~936 GB/s)")
    ap.add_argument("--peak-GFLOPs", type=float, default=35600.0,
                    help="peak compute for the roofline (default: RTX 3090 fp32 ~35.6 TFLOP/s)")
    ap.add_argument("--gpu-name", default="RTX 3090",
                    help="label written into roofline.json hardware block")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
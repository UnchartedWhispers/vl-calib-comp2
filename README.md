# VL-Calibration Visual-Certainty Kernel

GPU optimization project for **Computing for Data Science 2**. We take the
**token entropy** and **KL-divergence** computations that define VL-Calibration's
intrinsic visual certainty and benchmark three implementations of the dense,
memory-bound `[T, V]` kernel they reduce to:

1. **Eager** (naive multi-pass PyTorch) — the baseline.
2. **torch.compile** — same source, Inductor fuses the passes.
3. **Fused Triton** — single streaming pass (added separately).

We profile all three (Nsight), validate numerical agreement, and place them on a
roofline to show fusion cutting DRAM traffic.

---

## Baseline provenance (read this — it makes the project honest)

The upstream VL-Calibration paper (arXiv:2604.09529) defines visual certainty as
two logit-level quantities over the vocabulary `V`:

- **Internal certainty (entropy):** `H_t = -Σ_v p_t[v] log p_t[v]`, `p = softmax(logits_orig[t])`
- **Visual grounding (KL):** `KL_t = Σ_v p_t[v] (log p_t[v] - log q_t[v])`,
  `p = softmax(logits_orig[t])`, `q = softmax(logits_pert[t])` where the perturbed
  distribution comes from a perturbed input image.

In the released pipeline these are **collapsed to per-token scalars upstream**
(`log_probs_from_logits` in `dp_actor.py`), and `decouple.py` only shuffles those
scalars — no dense `[T, V]` tensor survives, so there is nothing GPU-relevant to
profile there. **We therefore reconstruct and benchmark the dense `[T, V]`
formulation the metric is mathematically defined on**, since that is the
memory-bound kernel worth fusing. All three arms compute exactly the same math on
identical inputs for an apples-to-apples comparison. State this plainly in the
report; do not imply we optimized a pre-existing GPU kernel from the repo.

---

## The data: you GENERATE it, you don't download it

The dense `[T, V]` logits never exist on disk in any dataset — they are transient
generation-time tensors. The `train_*_pipeline.jsonl` file only has
`problem / answer / ground_truth / images` plus **null placeholder** fields
(`response`, `vision_entropy`, `vision_kl`, ...). So the benchmark input is
produced by us, in two tiers:

- **Tier 1 — synthetic logits (PRIMARY benchmark).** `scripts/gen_synthetic.py`
  makes `[T, V]` logit pairs at the model's real vocab with realistic peakiness.
  A roofline study depends on tensor shape/dtype, not semantic content. Fully
  reproducible from a seed → **not committed to git**, regenerated on demand.
- **Tier 2 — real logits (correctness/realism anchor).** `scripts/dump_real_logits.py`
  runs the real VLM on ~16 samples (original + perturbed image) and dumps `[T, V]`
  logits. Proves the kernel gives sane entropy/KL on real data. GPU + weights
  required. Large → **not committed to git**.

The only data file committed is `data/real/train_32_pipeline.jsonl` (small, fixed
input; images are embedded as bytes so no separate image files are needed).

---

## Model

The paper applies VL-Calibration on Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct,
and InternVL3.5-4B-MPO. For the Tier-2 anchor we use the **smallest**,
`Qwen/Qwen3-VL-4B-Instruct` (vocab `151936`, bf16). The 4B is plenty for a
16-sample correctness check and kind to a single 3090. An FP8 variant
(`Qwen/Qwen3-VL-4B-Instruct-FP8`) exists if memory is tight.

**Unverified — confirm before the final Tier-2 run:** the exact image
**perturbation** the paper uses for the KL (blur vs noise vs masking). Check the
paper appendix / upstream code and match it in `dump_real_logits.py` (`--perturb`).
This does NOT affect Tier 1 or the roofline — the kernel computes KL over whatever
logit pairs it is given.

---

## Project layout

```
vlcalib_kernel/
  vc_kernel/
    ops.py              # vc_eager (baseline) + vc_compile (torch.compile)
    __init__.py
  scripts/
    gen_synthetic.py    # Tier 1: synthetic [T,V] logit pairs
    dump_real_logits.py # Tier 2: real logits from the VLM
    check_vocab.py      # read a model's vocab_size from config only
  data/
    real/
      train_32_pipeline.jsonl   # committed input (images embedded as bytes)
  results/              # gitignored: profiling + benchmark outputs
```

---

## A. Local setup (CPU, no GPU) — verify structure before the server

```bash
git clone <YOUR_REPO_URL> vlcalib_kernel
cd vlcalib_kernel

python -m venv .venv && source .venv/bin/activate
pip install torch pillow                 # CPU torch is fine locally

# Quick synthetic smoke test (tiny shapes, fp32 on CPU)
python -m scripts.gen_synthetic --T 64 --vocab 2000 --dtype fp32 --out data/synth

# Sanity-check the kernel produces valid entropy/KL
python -c "import torch; from vc_kernel.ops import vc_eager, reduce_sample; \
d=torch.load('data/synth/synth_T64_V2000_fp32.pt'); \
e,k=vc_eager(d['logits_orig'],d['logits_pert']); \
print('entropy<logV:', bool((e<=torch.log(torch.tensor(2000.))).all()), \
'| kl>=0:', bool((k>=-1e-4).all()), '| scalars:', reduce_sample(e,k))"
```

Expected: `entropy<logV: True | kl>=0: True | scalars: (...)`.

---

## B. GPU server runbook — follow top to bottom

### B0. Clone and environment

```bash
git clone <YOUR_REPO_URL> vlcalib_kernel
cd vlcalib_kernel

python -m venv .venv && source .venv/bin/activate

# Install the PyTorch build matching the server's CUDA (check `nvidia-smi`).
# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Qwen3-VL needs a recent transformers (>=4.57). triton ships with GPU torch.
pip install "transformers>=4.57" accelerate pillow torchvision qwen-vl-utils

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"
```

### B1. Confirm the model vocab, then generate Tier-1 synthetic data

```bash
# Print the vocab size from config (no weight download):
python -m scripts.check_vocab --model Qwen/Qwen3-VL-4B-Instruct
# -> Qwen/Qwen3-VL-4B-Instruct vocab_size = 151936

# Generate the roofline sweep with that vocab, bf16 (matches paper precision):
python -m scripts.gen_synthetic \
  --T 256 512 1024 2048 4096 \
  --vocab 151936 \
  --dtype bf16 \
  --seed 0 \
  --out data/synth
# Writes data/synth/synth_T{256..4096}_V151936_bf16.pt
```

If you switch models, rerun `check_vocab` and pass the new `--vocab`.

### B2. Dump Tier-2 real logits (start tiny, then scale)

```bash
# Smoke run first — 4 samples, short responses, to confirm the pipeline works:
python -m scripts.dump_real_logits \
  --jsonl data/real/train_32_pipeline.jsonl \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --n 4 --max-new-tokens 32 --perturb gaussian_blur \
  --out data/real

# Full anchor set once the smoke run is clean:
python -m scripts.dump_real_logits \
  --jsonl data/real/train_32_pipeline.jsonl \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --n 16 --max-new-tokens 128 --perturb gaussian_blur \
  --out data/real
# Writes data/real/real_000.pt ... real_015.pt
```

Memory note: bf16 logits are ~`V * 2` bytes per token (~0.3 MB/token at
V=151936). Keep `--n` and `--max-new-tokens` modest.

### B3. Validate the kernel on real logits

```bash
python -c "import torch,glob; from vc_kernel.ops import vc_eager, reduce_sample; \
f=sorted(glob.glob('data/real/real_*.pt'))[0]; d=torch.load(f); \
e,k=vc_eager(d['logits_orig'].cuda(), d['logits_pert'].cuda()); \
print(f, 'T=',d['T'],'| vision_entropy,vision_kl =', reduce_sample(e,k))"
```

Sane output = non-negative KL and entropy below `log(V)=~11.93`.

### B4. Benchmark + profile (next files, consume the .pt above)

The benchmark driver and Triton kernel are added separately. They will:
load `data/synth/*.pt`, time `vc_eager` / `vc_compile` / Triton with warmup +
CUDA-event timing, verify all three agree numerically, and emit roofline /
speedup data into `results/`. Profile with:

```bash
nsys profile -o results/vc_eager  python -m scripts.bench --arm eager  --T 4096
ncu --set full -o results/vc_triton python -m scripts.bench --arm triton --T 4096
```

---

## Git workflow

### First push (from your machine, after the local smoke test in A)

```bash
cd vlcalib_kernel
git init
git add .gitignore README.md vc_kernel/ scripts/ data/real/train_32_pipeline.jsonl
git status            # confirm NO *.pt and NO data/synth/ are staged
git commit -m "Visual-certainty kernel scaffold: eager+compile, data generators, runbook"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

### What is and isn't tracked

- **Committed:** code (`vc_kernel/`, `scripts/`), `README.md`, `.gitignore`, and the
  small fixed input `data/real/train_32_pipeline.jsonl`.
- **Ignored (regenerable / large):** `data/synth/*.pt`, `data/real/*.pt`,
  `results/`, profiler reports. Reproduce synthetic data from
  `gen_synthetic.py` + the seed; that command IS the reproducibility contract.

### On the GPU server

```bash
git pull                       # get latest code
# ... run B1-B4, which write only gitignored artifacts ...
git add scripts/bench.py vc_kernel/triton_kernel.py   # commit CODE you add
git commit -m "Add Triton kernel and benchmark driver"
git push
```

Never `git add data/synth` or `git add *.pt`. If you accidentally stage one:
`git restore --staged path/to/file.pt`.

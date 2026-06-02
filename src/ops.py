"""Visual-certainty kernels: token entropy and KL(p_orig || p_pert) over the full vocab.

VL-Calibration defines intrinsic visual certainty from two logit-level signals:
  * Internal certainty (entropy):  H_t = -sum_v p_t[v] log p_t[v],  p = softmax(logits_orig[t])
  * Visual grounding (KL):         KL_t = sum_v p_t[v] (log p_t[v] - log q_t[v]),
                                   p = softmax(logits_orig[t]), q = softmax(logits_pert[t])

Per sample these per-token values are mean-reduced over the visual-description
token span into the scalars `vision_entropy` / `vision_kl` consumed downstream.

BASELINE PROVENANCE: the released repo (decouple.py) consumes these as scalars;
log_probs_from_logits in dp_actor.py collapses [T,V] logits to [T] upstream.
We reconstruct and benchmark the DENSE [T,V] formulation the metric is defined
on, since that is the memory-bound kernel worth fusing. All three arms compute
exactly this, on identical inputs, for an apples-to-apples roofline comparison.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def vc_eager(logits_orig: torch.Tensor,
             logits_pert: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive multi-pass reference. Returns (entropy_per_token [T], kl_per_token [T]) fp32.

    Deliberately the obvious PyTorch way: ~5 separate passes over [T,V],
    each memory-bound. This is the DRAM-traffic baseline we want to cut.
    """
    lo = logits_orig.float()
    lp = logits_pert.float()
    logp = F.log_softmax(lo, dim=-1)      # pass 1 (+reduction)
    p = logp.exp()                        # pass 2
    entropy = -(p * logp).sum(dim=-1)     # pass 3 (+reduction)
    logq = F.log_softmax(lp, dim=-1)      # pass 4 (+reduction)
    kl = (p * (logp - logq)).sum(dim=-1)  # pass 5 (+reduction)
    return entropy, kl


# torch.compile arm: same source, let Inductor fuse the passes.
vc_compile = torch.compile(vc_eager, mode="max-autotune", fullgraph=True)


def reduce_sample(entropy_per_token: torch.Tensor,
                  kl_per_token: torch.Tensor) -> tuple[float, float]:
    """Mean-reduce per-token values to the per-sample scalars (vision_entropy, vision_kl)."""
    return float(entropy_per_token.mean()), float(kl_per_token.mean())

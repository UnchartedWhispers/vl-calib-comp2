"""Fused Triton kernel for the visual-certainty reduction.

The eager baseline in ops.py makes ~5 memory-bound passes over the [T, V]
logit pair:

    log_softmax(orig) -> p=exp -> entropy=-sum(p*logp)
    log_softmax(pert) -> kl=sum(p*(logp-logq))

Each pass re-reads [T, V] from DRAM. For a memory-bound kernel that is the whole
cost. This kernel fuses everything into a SINGLE streaming pass per row:

    one program == one token row t
    streaming pass A:  max_o, max_p          (numerically-stable maxima)
    streaming pass B:  sum exp(orig-max_o), sum exp(pert-max_p)  (logsumexp denoms)
    streaming pass C:  accumulate entropy and KL directly

We keep three passes over the row, but the row lives in registers/L2 across them
and each [T, V] element is read from DRAM exactly once per logits tensor (orig,
pert) rather than ~5x. That is the DRAM-traffic cut the roofline is meant to show.

Math (per token t, over vocab v):
    p[v]   = softmax(orig)[v]
    logp   = orig[v] - lse_o          lse_o = log sum_v exp(orig[v])
    logq   = pert[v] - lse_p          lse_p = log sum_v exp(pert[v])
    H_t    = -sum_v p[v] * logp[v]   = lse_o - sum_v p[v]*orig[v]
    KL_t   =  sum_v p[v] * (logp - logq)
           =  sum_v p[v] * ((orig[v]-lse_o) - (pert[v]-lse_p))
           =  (lse_p - lse_o) + sum_v p[v]*(orig[v] - pert[v])

All accumulation is done in fp32 regardless of input dtype, matching vc_eager,
which casts to float() before reducing.
"""
from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # CPU-only / no-triton environment: import still succeeds
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _vc_fused_kernel(
        orig_ptr, pert_ptr,            # *[T, V] input logits
        ent_ptr, kl_ptr,              # *[T]     fp32 outputs
        V,                            # vocab size (runtime)
        stride_t,                     # row stride for orig/pert (in elements)
        BLOCK: tl.constexpr,          # vocab tile width
    ):
        row = tl.program_id(0)
        orig_row = orig_ptr + row * stride_t
        pert_row = pert_ptr + row * stride_t

        NEG_INF = float("-inf")

        # ---- Pass A: row maxima for numerically-stable logsumexp ----
        m_o = NEG_INF
        m_p = NEG_INF
        for start in range(0, V, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < V
            o = tl.load(orig_row + offs, mask=mask, other=NEG_INF).to(tl.float32)
            p = tl.load(pert_row + offs, mask=mask, other=NEG_INF).to(tl.float32)
            m_o = tl.maximum(m_o, tl.max(o, axis=0))
            m_p = tl.maximum(m_p, tl.max(p, axis=0))

        # ---- Pass B: denominators sum exp(x - max) ----
        s_o = 0.0
        s_p = 0.0
        for start in range(0, V, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < V
            o = tl.load(orig_row + offs, mask=mask, other=NEG_INF).to(tl.float32)
            p = tl.load(pert_row + offs, mask=mask, other=NEG_INF).to(tl.float32)
            s_o += tl.sum(tl.where(mask, tl.exp(o - m_o), 0.0), axis=0)
            s_p += tl.sum(tl.where(mask, tl.exp(p - m_p), 0.0), axis=0)

        lse_o = m_o + tl.log(s_o)     # log sum exp(orig)
        lse_p = m_p + tl.log(s_p)     # log sum exp(pert)

        # ---- Pass C: accumulate weighted sums with p = softmax(orig) ----
        # H_t  = lse_o - sum_v p*orig
        # KL_t = (lse_p - lse_o) + sum_v p*(orig - pert)
        sum_p_orig = 0.0
        sum_p_diff = 0.0
        for start in range(0, V, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < V
            o = tl.load(orig_row + offs, mask=mask, other=0.0).to(tl.float32)
            p = tl.load(pert_row + offs, mask=mask, other=0.0).to(tl.float32)
            prob = tl.where(mask, tl.exp(o - lse_o), 0.0)      # softmax(orig)
            sum_p_orig += tl.sum(prob * o, axis=0)
            sum_p_diff += tl.sum(prob * (o - p), axis=0)

        entropy = lse_o - sum_p_orig
        kl = (lse_p - lse_o) + sum_p_diff

        tl.store(ent_ptr + row, entropy)
        tl.store(kl_ptr + row, kl)


def vc_triton(logits_orig: torch.Tensor,
              logits_pert: torch.Tensor,
              block: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused single-stream visual-certainty kernel.

    Returns (entropy_per_token [T], kl_per_token [T]) in fp32, matching vc_eager.
    Inputs must be CUDA tensors of shape [T, V]; orig and pert share a shape.
    """
    if not HAVE_TRITON:
        raise RuntimeError("Triton is not available in this environment (CPU-only).")
    assert logits_orig.is_cuda and logits_pert.is_cuda, "Triton arm requires CUDA tensors"
    assert logits_orig.shape == logits_pert.shape, "orig/pert shape mismatch"
    assert logits_orig.dim() == 2, "expected [T, V]"

    lo = logits_orig.contiguous()
    lp = logits_pert.contiguous()
    T, V = lo.shape

    entropy = torch.empty(T, device=lo.device, dtype=torch.float32)
    kl = torch.empty(T, device=lo.device, dtype=torch.float32)

    grid = (T,)
    _vc_fused_kernel[grid](
        lo, lp,
        entropy, kl,
        V,
        lo.stride(0),
        BLOCK=block,
    )
    return entropy, kl
import pytest
import torch
import torch.nn.functional as F
import torch_npu
import triton
import triton.language as tl
from einops import repeat


@triton.jit(do_not_specialize=["B", "T", "H", "HV"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    B: tl.int64,
    T: tl.int64,
    H: tl.int64,
    HV: tl.int64,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    all_tokens = B * T
    seq_len = eos - bos
    if seq_len == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_a = a + bos * HV + i_hv
    p_o = o + ((i_k * all_tokens + bos) * HV + i_hv) * V + o_v

    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    state_idx = tl.load(h0_indices + i_n).to(tl.int64)
    safe_state_idx = tl.where(state_idx < 0, 0, state_idx)
    p_h0 = (
        h0_source
        + safe_state_idx * HV * K * V
        + i_hv * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    b_h = tl.where(state_idx < 0, tl.zeros_like(b_h), b_h)

    for i_t in range(0, seq_len):
        b_q = tl.load(p_q + i_t * H * K, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k + i_t * H * K, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v + i_t * HV * V, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b + i_t * HV).to(tl.float32)
        b_a = tl.load(p_a + i_t * HV).to(tl.float32)

        x = b_a + tl.load(p_dt_bias).to(tl.float32)
        beta_x = softplus_beta * x
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)

        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(
            p_o + i_t * HV * V,
            b_o.to(p_o.dtype.element_ty),
            mask=mask_v,
        )

    if state_idx >= 0:
        p_h0 = (
            h0_source
            + state_idx * HV * K * V
            + i_hv * K * V
            + o_k[:, None] * V
            + o_v[None, :]
        )
        tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
):
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]
    N = cu_seqlens.numel() - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 64)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    if scale is None:
        scale = K**-0.5

    o = q.new_empty(NK, *v.shape)
    grid = (NK, NV, N * HV)
    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log,
        a,
        dt_bias,
        softplus_beta,
        softplus_threshold,
        q,
        k,
        v,
        b,
        o,
        initial_state_source,
        initial_state_indices,
        cu_seqlens,
        scale,
        B,
        T,
        H,
        HV,
        K,
        V,
        BK,
        BV,
        use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )
    return o.squeeze(0)


def torch_sigmoid_gating_delta_rule_update(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    initial_state,
    state_indices,
    cu_seqlens,
    scale,
    use_qk_l2norm_in_kernel,
):
    B, T, H, K = k.shape
    HV = v.shape[2]
    q_ref = repeat(q, "b t h d -> b t (h g) d", g=HV // H)
    k_ref = repeat(k, "b t h d -> b t (h g) d", g=HV // H)
    if use_qk_l2norm_in_kernel:
        q_ref = F.normalize(q_ref, p=2, dim=-1, eps=1e-6).to(q.dtype)
        k_ref = F.normalize(k_ref, p=2, dim=-1, eps=1e-6).to(k.dtype)

    out = torch.empty((B, T, HV, v.shape[-1]), dtype=q.dtype)
    next_state = initial_state.clone()
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias, beta=1.0, threshold=20.0)
    beta = b.sigmoid()
    for seq_idx in range(cu_seqlens.numel() - 1):
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        state_idx = int(state_indices[seq_idx].item())
        h = torch.zeros_like(next_state[0]) if state_idx < 0 else next_state[state_idx].clone()
        for token_idx in range(bos, eos):
            batch_idx = token_idx // T
            local_idx = token_idx % T
            h = h * torch.exp(g.reshape(B * T, HV)[token_idx]).view(HV, 1, 1)
            v_cur = v.reshape(B * T, HV, -1)[token_idx].float()
            k_cur = k_ref.reshape(B * T, HV, K)[token_idx].float()
            q_cur = q_ref.reshape(B * T, HV, K)[token_idx].float() * scale
            v_cur = v_cur - torch.sum(h * k_cur[:, :, None], dim=1)
            v_cur = v_cur * beta.reshape(B * T, HV)[token_idx].float()[:, None]
            h = h + k_cur[:, :, None] * v_cur[:, None, :]
            out[batch_idx, local_idx] = torch.sum(h * q_cur[:, :, None], dim=1).to(q.dtype)
        if state_idx >= 0:
            next_state[state_idx] = h
    return out, next_state


@pytest.mark.parametrize(
    "batch,H,HV",
    [
        (1, 8, 16),
        (2, 8, 16),
        (4, 8, 16),
        (8, 8, 16),
        # Qwen3.5/Qwen3.6 local GDN shapes for TP1/2/4/8.
        (4, 16, 16),
        (4, 8, 8),
        (4, 4, 4),
        (4, 2, 2),
        (4, 16, 32),
        (4, 4, 8),
        (4, 2, 4),
        (4, 16, 48),
        (4, 8, 24),
        (4, 4, 12),
        (4, 2, 6),
        (4, 16, 64),
        (4, 8, 32),
        (4, 4, 16),
        (4, 2, 8),
        # Regression for the previously observed TP4 spec-local shape.
        (4, 1, 8),
    ],
)
def test_fused_sigmoid_gating_delta_rule_update_dynamic_batch(batch, H, HV):
    device = "npu"
    torch.manual_seed(47 + batch + H + HV)
    T, K, V = 1, 128, 128
    dtype = torch.bfloat16
    scale = K**-0.5
    q = torch.randn(batch, T, H, K, dtype=dtype)
    k = torch.randn(batch, T, H, K, dtype=dtype)
    v = torch.randn(batch, T, HV, V, dtype=dtype)
    a = torch.randn(batch, T, HV, dtype=dtype)
    b = torch.randn(batch, T, HV, dtype=dtype)
    A_log = torch.randn(HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    initial_state = torch.randn(batch, HV, K, V, dtype=torch.float32)
    state_indices = torch.arange(batch, dtype=torch.int32)
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32) * T

    ref_out, ref_state = torch_sigmoid_gating_delta_rule_update(
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        initial_state,
        state_indices,
        cu_seqlens,
        scale,
        True,
    )

    state_d = initial_state.to(device)
    tri_out = fused_sigmoid_gating_delta_rule_update(
        A_log.to(device),
        a.to(device),
        dt_bias.to(device),
        q.to(device),
        k.to(device),
        v.to(device),
        b.to(device),
        state_d,
        state_indices.to(device),
        cu_seqlens.to(device),
        scale,
        True,
    )
    torch.testing.assert_close(
        ref_out.float(), tri_out.cpu().float(), rtol=0.005, atol=0.01
    )
    torch.testing.assert_close(
        ref_state.float(), state_d.cpu().float(), rtol=0.05, atol=0.02
    )

import os
from typing import Optional, Union

import pytest
import torch
import torch.nn.functional as F
import torch_npu
import triton
import triton.language as tl


PAD_SLOT_ID = -1
RUN_EXTENDED_V2 = os.getenv("CAUSAL_CONV1D_V2_EXTENDED") == "1"


@triton.jit(
    do_not_specialize=[
        "batch",
        "dim",
        "seqlen",
        "state_len",
        "num_cache_lines",
        "stride_x_token",
        "stride_w_width",
        "stride_conv_state_seq",
        "stride_conv_state_tok",
        "stride_state_indices",
        "stride_o_token",
        "kernel_width",
        "is_spec_decoding",
        "pad_slot_id",
    ]
)
def _causal_conv1d_update_kernel_npu_tiled_v2(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,
    block_idx_last_scheduled_token_ptr,
    initial_state_idx_ptr,
    o_ptr,
    batch: tl.int32,
    dim,
    seqlen,
    state_len,
    num_cache_lines,
    stride_x_dim: tl.constexpr,
    stride_x_token,
    stride_w_dim: tl.constexpr,
    stride_w_width,
    stride_conv_state_seq,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok,
    stride_state_indices,
    stride_o_dim: tl.constexpr,
    stride_o_token,
    kernel_width,
    is_spec_decoding,
    pad_slot_id,
    BLOCK_N: tl.constexpr,
    B_TILE: tl.constexpr,
    T_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim

    kw = kernel_width.to(tl.int32)
    spec_enabled = is_spec_decoding.to(tl.int32) != 0

    w_base = w_ptr + idx_feats * stride_w_dim
    w_col0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w & (kw > 0), other=0.0).to(tl.float32)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w & (kw > 1), other=0.0).to(tl.float32)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w & (kw > 2), other=0.0).to(tl.float32)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w & (kw > 3), other=0.0).to(tl.float32)
    w_col4 = tl.load(w_base + 4 * stride_w_width, mask=mask_w & (kw > 4), other=0.0).to(tl.float32)
    w_col5 = tl.load(w_base + 5 * stride_w_width, mask=mask_w & (kw > 5), other=0.0).to(tl.float32)

    acc_bias = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)

    tok_vec = tl.arange(0, T_CHUNK)

    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi
        lane_active = b < batch

        conv_state_init = tl.load(initial_state_idx_ptr + b, mask=lane_active, other=0).to(tl.int32)
        current_last_index = tl.load(block_idx_last_scheduled_token_ptr + b, mask=lane_active, other=0).to(tl.int32)

        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + conv_state_init,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        lane_active = lane_active & ((pad_slot_id < 0) | (conv_states_input_coord != pad_slot_id))

        qs = tl.load(query_start_loc_ptr + b, mask=lane_active, other=0).to(tl.int64)
        qe = tl.load(query_start_loc_ptr + (b + 1), mask=lane_active, other=0).to(tl.int64)
        seqlen_run = (qe - qs).to(tl.int32)
        state_len_run = (state_len - (seqlen - seqlen_run)).to(tl.int32)
        x_offset = (qs * stride_x_token).to(tl.int64)
        o_offset = (qs * stride_o_token).to(tl.int64)

        lane_active = lane_active & (seqlen_run > 0)

        accepted_tokens = tl.load(num_accepted_tokens_ptr + b, mask=lane_active, other=1).to(tl.int64)
        conv_state_token_offset = tl.where(spec_enabled, accepted_tokens - 1, 0).to(tl.int64)
        shift = tl.where(spec_enabled, 1, seqlen_run).to(tl.int32)

        conv_states_base = (
            conv_state_ptr + conv_states_input_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim
        )
        prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok

        col0 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col1 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col2 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col3 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col4 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col0 = tl.load(
            prior_tokens + 0 * stride_conv_state_tok,
            mask=lane_active & mask_w & (kw > 1),
            other=0.0,
        ).to(tl.float16)
        col1 = tl.load(
            prior_tokens + 1 * stride_conv_state_tok,
            mask=lane_active & mask_w & (kw > 2),
            other=0.0,
        ).to(tl.float16)
        col2 = tl.load(
            prior_tokens + 2 * stride_conv_state_tok,
            mask=lane_active & mask_w & (kw > 3),
            other=0.0,
        ).to(tl.float16)
        col3 = tl.load(
            prior_tokens + 3 * stride_conv_state_tok,
            mask=lane_active & mask_w & (kw > 4),
            other=0.0,
        ).to(tl.float16)
        col4 = tl.load(
            prior_tokens + 4 * stride_conv_state_tok,
            mask=lane_active & mask_w & (kw > 5),
            other=0.0,
        ).to(tl.float16)

        conv_states_offset = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + current_last_index,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        use_shift = seqlen_run < state_len_run
        use_tail = seqlen_run >= state_len_run
        keep_shift = tl.where(use_shift, (state_len_run - seqlen_run), 0).to(tl.int32)
        tail_start = tl.where(use_tail, (seqlen_run - state_len_run), 0).to(tl.int32)

        state_src_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + conv_state_token_offset * stride_conv_state_tok
            + idx_feats * stride_conv_state_dim
        )
        state_dst_base = conv_state_ptr + conv_states_offset * stride_conv_state_seq + idx_feats * stride_conv_state_dim
        x_base = x_ptr + x_offset + idx_feats * stride_x_dim

        for t0 in tl.range(0, state_len_run, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)
            src_tok = (dst_tok + shift).to(tl.int32)
            m_tok = use_shift & (dst_tok < keep_shift) & (src_tok < state_len_run) & (dst_tok < state_len_run)
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_input_coord < num_cache_lines)
                & (conv_states_offset < num_cache_lines)
            )
            src_ptrs = state_src_base[None, :] + src_tok[:, None] * stride_conv_state_tok
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            vals = tl.load(src_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, vals, mask=m)

        for t0 in tl.range(0, seqlen_run, T_CHUNK):
            x_tok = (t0 + tok_vec).to(tl.int32)
            dst_tok = (keep_shift + x_tok).to(tl.int32)
            m_tok = use_shift & (x_tok < seqlen_run) & (dst_tok < state_len_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (conv_states_offset < num_cache_lines)
            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        for t0 in tl.range(0, state_len_run, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)
            x_tok = (tail_start + dst_tok).to(tl.int32)
            m_tok = use_tail & (dst_tok < state_len_run) & (x_tok < seqlen_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (conv_states_offset < num_cache_lines)
            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        x_base_1d = x_base
        o_base_1d = o_ptr + o_offset + idx_feats * stride_o_dim

        for idx_token in tl.range(seqlen_run):
            acc = acc_bias
            x_ptrs_1d = x_base_1d + idx_token * stride_x_token
            current_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)

            acc += tl.where(kw > 1, col0.to(tl.float32) * w_col0, 0.0)
            acc += tl.where(kw > 2, col1.to(tl.float32) * w_col1, 0.0)
            acc += tl.where(kw > 3, col2.to(tl.float32) * w_col2, 0.0)
            acc += tl.where(kw > 4, col3.to(tl.float32) * w_col3, 0.0)
            acc += tl.where(kw > 5, col4.to(tl.float32) * w_col4, 0.0)

            current_w = (
                tl.where(kw == 1, w_col0, 0.0)
                + tl.where(kw == 2, w_col1, 0.0)
                + tl.where(kw == 3, w_col2, 0.0)
                + tl.where(kw == 4, w_col3, 0.0)
                + tl.where(kw == 5, w_col4, 0.0)
                + tl.where(kw == 6, w_col5, 0.0)
            )
            acc += current_x.to(tl.float32) * current_w

            old0 = col0
            old1 = col1
            old2 = col2
            old3 = col3
            old4 = col4
            col0 = tl.where(kw > 1, tl.where(kw == 2, current_x, old1), col0)
            col1 = tl.where(kw > 2, tl.where(kw == 3, current_x, old2), col1)
            col2 = tl.where(kw > 3, tl.where(kw == 4, current_x, old3), col2)
            col3 = tl.where(kw > 4, tl.where(kw == 5, current_x, old4), col3)
            col4 = tl.where(kw > 5, current_x, col4)

            o_ptrs = o_base_1d + idx_token * stride_o_token
            tl.store(o_ptrs, acc, mask=lane_active & mask_w)


def _resolve_cache_index(conv_state_indices: Optional[torch.Tensor], batch_idx: int, slot_idx: int) -> int:
    if conv_state_indices is None:
        return batch_idx
    if conv_state_indices.dim() == 1:
        return int(conv_state_indices[batch_idx].item())
    return int(conv_state_indices[batch_idx, slot_idx].item())


def causal_conv1d_update_ref_v2(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
):
    if isinstance(activation, bool):
        activation = "silu" if activation else None

    original_dtype = x.dtype
    if query_start_loc is None:
        x_work = x.unsqueeze(-1) if x.dim() == 2 else x
        batch, dim, _ = x_work.shape
        out = torch.empty_like(x_work)
        seqlen = x_work.shape[-1]
    else:
        batch = conv_state_indices.size(0)
        dim = x.shape[1]
        out = torch.empty_like(x)
        seqlen = max_query_len

    width = weight.shape[1]
    eff_state_len = width - 1 + (seqlen - 1 if num_accepted_tokens is not None else 0)

    for b in range(batch):
        init_slot = int(initial_state_idx[b].item()) if initial_state_idx is not None else 0
        last_slot = int(block_idx_last_scheduled_token[b].item()) if block_idx_last_scheduled_token is not None else 0

        in_idx = _resolve_cache_index(conv_state_indices, b, init_slot)
        if in_idx == pad_slot_id:
            continue
        out_idx = _resolve_cache_index(conv_state_indices, b, last_slot)

        if query_start_loc is None:
            seq = x_work[b]
            seq_len = seq.shape[-1]
            state_len_run = eff_state_len
        else:
            start = int(query_start_loc[b].item())
            end = int(query_start_loc[b + 1].item())
            seq = x[start:end].transpose(0, 1)
            seq_len = end - start
            state_len_run = eff_state_len - (seqlen - seq_len)

        if seq_len == 0:
            continue

        token_offset = int(num_accepted_tokens[b].item()) - 1 if num_accepted_tokens is not None else 0
        history = conv_state[in_idx, :, token_offset : token_offset + width - 1]
        src_state = conv_state[in_idx, :, token_offset : token_offset + state_len_run]

        cat_x = torch.cat([history, seq], dim=-1).to(weight.dtype).unsqueeze(0)
        out_seq = F.conv1d(cat_x, weight.unsqueeze(1), bias, padding=0, groups=dim)[:, :, -seq_len:]
        if activation in ["silu", "swish"]:
            out_seq = F.silu(out_seq)

        if query_start_loc is None:
            out[b] = out_seq.squeeze(0).to(original_dtype)
        else:
            out[start:end] = out_seq.squeeze(0).transpose(0, 1).to(original_dtype)

        if seq_len >= state_len_run:
            new_state = seq[:, -state_len_run:]
        else:
            shift = 1 if num_accepted_tokens is not None else seq_len
            keep = state_len_run - seq_len
            new_state = torch.cat([src_state[:, shift : shift + keep], seq], dim=-1)
        conv_state[out_idx, :, :state_len_run].copy_(new_state.to(conv_state.dtype))

    if query_start_loc is None and x.dim() == 2:
        out = out.squeeze(-1)
    return out.to(original_dtype)


def causal_conv1d_update_npu_v2(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Union[bool, str, None] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    validate_data: bool = False,
):
    if isinstance(activation, bool):
        activation = "silu" if activation else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_dtype = x.dtype
    x_work = x.to(conv_state.dtype)
    unsqueeze = False

    if query_start_loc is None:
        unsqueeze = x_work.dim() == 2
        if unsqueeze:
            x_work = x_work.unsqueeze(-1)
        assert x_work.dim() == 3
        batch, dim, seqlen = x_work.shape
        x_kernel = x_work.transpose(1, 2).contiguous().view(batch * seqlen, dim)
        query_start_loc_kernel = torch.arange(
            0,
            (batch + 1) * seqlen,
            seqlen,
            device=conv_state.device,
            dtype=torch.int32,
        )
    else:
        assert x_work.dim() == 2
        query_start_loc_kernel = query_start_loc.to(torch.int32).contiguous()
        assert query_start_loc_kernel.dim() == 1 and query_start_loc_kernel.numel() >= 2
        batch = query_start_loc_kernel.numel() - 1
        dim = x_work.size(1)
        assert max_query_len > 0
        seqlen = max_query_len
        x_kernel = x_work.contiguous()

    if validate_data:
        assert weight.shape[0] == dim
        assert conv_state.shape[1] == dim
        assert 1 <= weight.shape[1] <= 6

    width = weight.shape[1]
    assert 1 <= width <= 6

    if bias is None:
        bias_kernel = torch.zeros(dim, device=conv_state.device, dtype=weight.dtype)
    else:
        bias_kernel = bias.contiguous()
        assert bias_kernel.shape == (dim,)

    if conv_state_indices is None:
        base_idx = torch.arange(batch, device=conv_state.device, dtype=torch.int32)
        conv_state_indices_kernel = torch.stack([base_idx, base_idx], dim=1).contiguous()
    else:
        conv_state_indices_kernel = conv_state_indices.to(torch.int32).contiguous()
        if conv_state_indices_kernel.dim() == 1:
            conv_state_indices_kernel = torch.stack(
                [conv_state_indices_kernel, conv_state_indices_kernel], dim=1
            ).contiguous()
        else:
            assert conv_state_indices_kernel.dim() == 2
            assert conv_state_indices_kernel.size(0) == batch
            assert conv_state_indices_kernel.size(1) >= 2

    weight_kernel = weight.transpose(0, 1).contiguous()
    conv_state_kernel = conv_state.transpose(1, 2).contiguous()
    out_kernel = torch.empty_like(x_kernel)

    num_cache_lines, _, _ = conv_state_kernel.shape
    eff_state_len = width - 1 + (seqlen - 1 if num_accepted_tokens is not None else 0)

    is_spec_decoding = 0
    if num_accepted_tokens is None:
        num_accepted_tokens_kernel = torch.ones(batch, device=conv_state.device, dtype=torch.int32)
    else:
        num_accepted_tokens_kernel = num_accepted_tokens.to(torch.int32).contiguous()
        is_spec_decoding = 1

    if block_idx_last_scheduled_token is None:
        block_idx_last_scheduled_token_kernel = torch.zeros(batch, device=conv_state.device, dtype=torch.int32)
    else:
        block_idx_last_scheduled_token_kernel = block_idx_last_scheduled_token.to(torch.int32).contiguous()

    if initial_state_idx is None:
        initial_state_idx_kernel = torch.zeros(batch, device=conv_state.device, dtype=torch.int32)
    else:
        initial_state_idx_kernel = initial_state_idx.to(torch.int32).contiguous()

    stride_w_width, stride_w_dim = weight_kernel.stride()
    stride_x_token, stride_x_dim = x_kernel.stride()
    stride_o_token, stride_o_dim = out_kernel.stride()
    stride_state_seq, stride_state_token, stride_state_dim = conv_state_kernel.stride()
    stride_state_indices = conv_state_indices_kernel.stride(0)

    assert stride_x_dim == 1
    assert stride_w_dim == 1
    assert stride_state_dim == 1
    assert stride_o_dim == 1

    def grid(meta):
        return (triton.cdiv(batch, meta["B_TILE"]), triton.cdiv(dim, meta["BLOCK_N"]))

    _causal_conv1d_update_kernel_npu_tiled_v2[grid](
        x_kernel,
        weight_kernel,
        bias_kernel,
        conv_state_kernel,
        conv_state_indices_kernel,
        num_accepted_tokens_kernel,
        query_start_loc_kernel,
        block_idx_last_scheduled_token_kernel,
        initial_state_idx_kernel,
        out_kernel,
        batch,
        dim,
        seqlen,
        eff_state_len,
        num_cache_lines,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_state_seq,
        stride_state_dim,
        stride_state_token,
        stride_state_indices,
        stride_o_dim,
        stride_o_token,
        width,
        is_spec_decoding,
        pad_slot_id,
        BLOCK_N=256,
        B_TILE=1,
        T_CHUNK=16,
    )

    conv_state.copy_(conv_state_kernel.transpose(1, 2))
    out = out_kernel
    if activation in ["silu", "swish"]:
        out = F.silu(out)

    if query_start_loc is None:
        out = out.view(batch, seqlen, dim).transpose(1, 2).contiguous()
        if unsqueeze:
            out = out.squeeze(-1)
    return out.to(original_dtype)


@pytest.mark.parametrize("bs", [1, 4])
@pytest.mark.parametrize("dim", [1024, 2048])
def test_causal_conv1d_update_v2_decode(bs, dim):
    device = "npu"
    dtype = torch.bfloat16
    width = 4
    seqlen = 1
    activation = "silu"

    x = torch.randn(bs, dim, seqlen, device=device, dtype=dtype)
    conv_state = torch.randn(bs, dim, width - 1, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.zeros(dim, device=device, dtype=dtype)
    # Compile the v2 kernel with APC enabled by using a 2D slot table where
    # both input/output slots resolve to the same cache line.
    conv_state_indices = torch.stack(
        [
            torch.arange(bs, device=device, dtype=torch.int32),
            torch.arange(bs, device=device, dtype=torch.int32),
        ],
        dim=1,
    ).contiguous()
    query_start_loc = torch.arange(bs + 1, device=device, dtype=torch.int32)
    block_idx_last_scheduled_token = torch.zeros(bs, device=device, dtype=torch.int32)
    initial_state_idx = torch.zeros(bs, device=device, dtype=torch.int32)

    x_varlen = x.squeeze(-1).contiguous()

    x_ref = x_varlen.cpu()
    conv_state_ref = conv_state.cpu()
    weight_ref = weight.cpu()
    bias_ref = bias.cpu()
    indices_ref = conv_state_indices.cpu()

    out = causal_conv1d_update_npu_v2(
        x_varlen,
        conv_state,
        weight,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        query_start_loc=query_start_loc,
        max_query_len=1,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
    )
    out_ref = causal_conv1d_update_ref_v2(
        x_ref,
        conv_state_ref,
        weight_ref,
        bias=bias_ref,
        activation=activation,
        conv_state_indices=indices_ref,
        query_start_loc=query_start_loc.cpu(),
        max_query_len=1,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token.cpu(),
        initial_state_idx=initial_state_idx.cpu(),
    )

    assert torch.equal(conv_state.cpu(), conv_state_ref)
    assert torch.allclose(out.cpu(), out_ref, rtol=1e-2, atol=5e-2)


def test_causal_conv1d_update_v2_dense_no_bias_no_activation_width2():
    device = "npu"
    dtype = torch.bfloat16
    batch = 2
    dim = 1024
    width = 2
    seqlen = 3

    x = torch.randn(batch, dim, seqlen, device=device, dtype=dtype)
    conv_state = torch.randn(batch, dim, width - 1, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)

    x_ref = x.cpu()
    conv_state_ref = conv_state.cpu()
    weight_ref = weight.cpu()

    out = causal_conv1d_update_npu_v2(
        x,
        conv_state,
        weight,
        bias=None,
        activation=None,
        query_start_loc=None,
    )
    out_ref = causal_conv1d_update_ref_v2(
        x_ref,
        conv_state_ref,
        weight_ref,
        bias=None,
        activation=None,
        query_start_loc=None,
    )

    assert torch.equal(conv_state.cpu(), conv_state_ref)
    assert torch.allclose(out.cpu(), out_ref, rtol=1e-2, atol=5e-2)


def test_causal_conv1d_update_v2_varlen_pad_width5():
    device = "npu"
    dtype = torch.bfloat16
    batch = 3
    dim = 1024
    width = 5
    seqlens = [3, 0, 2]
    activation = "silu"
    max_query_len = max(seqlens)
    total_tokens = sum(seqlens)

    x = torch.randn(total_tokens, dim, device=device, dtype=dtype)
    conv_state = torch.randn(4, dim, width - 1, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.zeros(dim, device=device, dtype=dtype)
    conv_state_indices = torch.tensor(
        [[0, 1], [-7, -7], [2, 3]],
        device=device,
        dtype=torch.int32,
    )
    block_idx_last_scheduled_token = torch.tensor([1, 0, 1], device=device, dtype=torch.int32)
    initial_state_idx = torch.tensor([0, 0, 0], device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 3, 5], device=device, dtype=torch.int32)

    x_ref = x.cpu()
    conv_state_ref = conv_state.cpu()
    weight_ref = weight.cpu()
    bias_ref = bias.cpu()

    out = causal_conv1d_update_npu_v2(
        x,
        conv_state,
        weight,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        query_start_loc=query_start_loc,
        max_query_len=max_query_len,
        pad_slot_id=-7,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
    )
    out_ref = causal_conv1d_update_ref_v2(
        x_ref,
        conv_state_ref,
        weight_ref,
        bias=bias_ref,
        activation=activation,
        conv_state_indices=conv_state_indices.cpu(),
        query_start_loc=query_start_loc.cpu(),
        max_query_len=max_query_len,
        pad_slot_id=-7,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token.cpu(),
        initial_state_idx=initial_state_idx.cpu(),
    )

    assert torch.equal(conv_state.cpu(), conv_state_ref)
    assert torch.allclose(out.cpu(), out_ref, rtol=1e-2, atol=5e-2)


@pytest.mark.skipif(
    not RUN_EXTENDED_V2,
    reason="Extended v2 coverage is disabled during default AOT collection.",
)
def test_causal_conv1d_update_v2_varlen():
    device = "npu"
    dtype = torch.bfloat16
    dim = 2048
    width = 4
    activation = "silu"

    seqlens = [2, 1]
    total_tokens = sum(seqlens)
    x = torch.randn(total_tokens, dim, device=device, dtype=dtype)
    conv_state = torch.randn(len(seqlens), dim, width - 1, device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    conv_state_indices = torch.arange(len(seqlens), device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, seqlens[0], total_tokens], device=device, dtype=torch.int32)

    x_ref = x.cpu()
    conv_state_ref = conv_state.cpu()
    weight_ref = weight.cpu()

    out = causal_conv1d_update_npu_v2(
        x,
        conv_state,
        weight,
        activation=activation,
        conv_state_indices=conv_state_indices,
        query_start_loc=query_start_loc,
        max_query_len=max(seqlens),
    )
    out_ref = causal_conv1d_update_ref_v2(
        x_ref,
        conv_state_ref,
        weight_ref,
        activation=activation,
        conv_state_indices=conv_state_indices.cpu(),
        query_start_loc=query_start_loc.cpu(),
        max_query_len=max(seqlens),
    )

    assert torch.equal(conv_state.cpu(), conv_state_ref)
    assert torch.allclose(out.cpu(), out_ref, rtol=1e-2, atol=5e-2)


@pytest.mark.skipif(
    not RUN_EXTENDED_V2,
    reason="Extended v2 coverage is disabled during default AOT collection.",
)
def test_causal_conv1d_update_v2_spec_decoding():
    device = "npu"
    dtype = torch.bfloat16
    batch = 2
    dim = 2048
    width = 4
    seqlen = 2
    activation = "silu"

    x = torch.randn(batch, dim, seqlen, device=device, dtype=dtype)
    conv_state = torch.randn(batch, dim, width - 1 + (seqlen - 1), device=device, dtype=dtype)
    weight = torch.randn(dim, width, device=device, dtype=dtype)
    conv_state_indices = torch.arange(batch, device=device, dtype=torch.int32)
    num_accepted_tokens = torch.tensor([1, 2], device=device, dtype=torch.int32)

    x_ref = x.cpu()
    conv_state_ref = conv_state.cpu()
    weight_ref = weight.cpu()

    out = causal_conv1d_update_npu_v2(
        x,
        conv_state,
        weight,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )
    out_ref = causal_conv1d_update_ref_v2(
        x_ref,
        conv_state_ref,
        weight_ref,
        activation=activation,
        conv_state_indices=conv_state_indices.cpu(),
        num_accepted_tokens=num_accepted_tokens.cpu(),
    )

    assert torch.equal(conv_state.cpu(), conv_state_ref)
    assert torch.allclose(out.cpu(), out_ref, rtol=1e-2, atol=5e-2)


if __name__ == "__main__":
    test_causal_conv1d_update_v2_decode(bs=1, dim=2048)

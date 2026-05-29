import torch
import triton
import triton.language as tl
import pytest
import numpy as np

DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-3
DEVICE = "npu:0"
TOTAL_NUM_Q_HEADS = [32, 64]
TOTAL_NUM_KV_HEADS = 8
SUPPORTED_TP_SIZES = (1, 2, 4, 8, 16)


def compute_tp_head_configs():
    head_configs = []
    for Q_HEADS in TOTAL_NUM_Q_HEADS:
        for tp_size in SUPPORTED_TP_SIZES:
            assert Q_HEADS % tp_size == 0
            num_q_heads = Q_HEADS // tp_size
            if TOTAL_NUM_KV_HEADS >= tp_size:
                assert TOTAL_NUM_KV_HEADS % tp_size == 0
                num_kv_heads = TOTAL_NUM_KV_HEADS // tp_size
            else:
                assert tp_size % TOTAL_NUM_KV_HEADS == 0
                num_kv_heads = 1
            head_configs.append((num_q_heads, num_kv_heads))
    return head_configs


def get_vectorcore_num():
    return 32


def custom_kernel_repr(proxy):
    kernel_name = proxy.fn.__name__
    repr_constants = (
        ("BIAS", "bias"),
        ("eps", "eps"),
        ("HEAD_DIM", "hd"),
        ("q_hidden_size", "qh"),
        ("kv_hidden_size", "kvh"),
    )
    constants_str = []
    for name, clean_name in repr_constants:
        if name not in proxy.constants:
            continue
        const_value = proxy.constants[name]
        if isinstance(const_value, bool):
            clean_value = 1 if const_value else 0
        elif isinstance(const_value, float):
            clean_value = format(const_value, 'f').replace('.', '').replace('-', '')
        else:
            clean_value = const_value
        constants_str.append(f"{clean_name}{clean_value}")
    if constants_str:
        return f"{kernel_name}_{'_'.join(constants_str)}"
    else:
        return kernel_name


@triton.jit(repr=custom_kernel_repr)
def split_rmsnorm_rope_kernel(
    input_ptr,
    sin_ptr,
    cos_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)
    weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
    input_offset = row_pid * total_hidden_size
    output_offset = row_pid * q_hidden_size
    input_offset_step = row_step * total_hidden_size
    output_offset_step = row_step * q_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
        valid_mask = col_indices < q_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(Q_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = input_values * reciprocal_std
        if BIAS:
            normalized_values = normalized_values * weight_values + bias_values
        else:
            normalized_values = normalized_values * weight_values

        sc_offsets = row_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        x1 = tl.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            normalized_values,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM), dtype=tl.float32)
        cat_x = tl.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        roped_q = cat_x * sin + normalized_values * cos
        tl.store(
            q_ptr + output_offset + col_indices,
            roped_q.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))
    input_offset = row_pid * total_hidden_size + q_hidden_size
    output_offset = row_pid * kv_hidden_size
    output_offset_step = row_step * kv_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(KV_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = input_values * reciprocal_std
        if BIAS:
            normalized_values = normalized_values * weight_values + bias_values
        else:
            normalized_values = normalized_values * weight_values

        sc_offsets = row_idx * HEAD_DIM + tl.arange(0, HEAD_DIM)
        sin = (tl.load(sin_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        cos = (tl.load(cos_ptr + sc_offsets)).reshape(1, HEAD_DIM)
        x1 = tl.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            normalized_values,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.zeros((KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM), dtype=tl.float32)
        cat_x = tl.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        cat_x = tl.insert_slice(
            cat_x,
            x1,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        roped_k = cat_x * sin + normalized_values * cos
        tl.store(
            k_ptr + output_offset + col_indices,
            roped_k.to(k_ptr.dtype.element_ty).reshape(KV_BLOCK_SIZE),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    input_offset = row_pid * total_hidden_size + q_hidden_size + kv_hidden_size
    output_offset = row_pid * kv_hidden_size
    for _ in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
        tl.store(v_ptr + output_offset + col_indices, input_values, mask=valid_mask)
        input_offset += input_offset_step
        output_offset += output_offset_step


def triton_split_rmsnorm_rope_impl(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert head_dim == KV_BLOCK_SIZE
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    n_cols = kv_hidden_size // KV_BLOCK_SIZE
    num_vectorcore = get_vectorcore_num()
    assert num_vectorcore % n_cols == 0
    n_rows = num_vectorcore // n_cols

    split_rmsnorm_rope_kernel[(n_rows, n_cols, 1)](
        input,
        sin,
        cos,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        batch_size,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        bias,
        head_dim,
        head_dim // 2,
    )
    return q_output, k_output, v_output


def custom_rope(q, k, sin, cos):
    rotary_dim = sin.shape[-1]
    sin = sin.to(torch.float32)
    cos = cos.to(torch.float32)

    while sin.dim() < 4:
        sin = sin.unsqueeze(1)
        cos = cos.unsqueeze(1)

    x1 = q[..., :rotary_dim // 2]
    x2 = q[..., rotary_dim // 2:]
    cat_x = torch.cat([-x2, x1], axis=-1)
    mul1 = cat_x * sin
    mul2 = q * cos
    res1 = mul1 + mul2

    x1 = k[..., :rotary_dim // 2]
    x2 = k[..., rotary_dim // 2:]
    cat_x = torch.cat([-x2, x1], axis=-1)
    mul1 = cat_x * sin
    mul2 = k * cos
    res2 = mul1 + mul2
    return res1, res2


def split_rmsnorm_rope_ref(
    input: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    bias: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _q, _k, v_gold = input.cpu().split(
        [q_hidden_size, kv_hidden_size, kv_hidden_size], dim=-1)
    if bias and q_bias is not None and k_bias is not None:
        _q = rms_norm(_q.reshape(-1, head_dim),
                      q_weight.cpu(),
                      eps,
                      norm_bias=q_bias.cpu())
        _k = rms_norm(_k.reshape(-1, head_dim),
                      k_weight.cpu(),
                      eps,
                      norm_bias=k_bias.cpu())
    else:
        _q = rms_norm(_q.reshape(-1, head_dim), q_weight.cpu(), eps)
        _k = rms_norm(_k.reshape(-1, head_dim), k_weight.cpu(), eps)

    num_tokens = input.shape[0]
    _q = _q.reshape(num_tokens, 1, -1, head_dim)
    _k = _k.reshape(num_tokens, 1, -1, head_dim)

    q_gold, k_gold = custom_rope(_q, _k, sin.cpu(), cos.cpu())
    q_gold = q_gold.reshape(num_tokens, -1)
    k_gold = k_gold.reshape(num_tokens, -1)

    return q_gold, k_gold, v_gold


def rms_norm(
    input,
    norm_weight,
    eps,
    norm_bias=None,
):
    input = input.to(torch.float32)
    norm_weight = norm_weight.to(torch.float32)
    reciprocal_std = 1 / torch.sqrt(
        torch.mean(input**2, axis=-1, keepdims=True) + eps)
    out = input * reciprocal_std * norm_weight
    if norm_bias is not None:
        norm_bias = norm_bias.to(torch.float32)
        out = out + norm_bias
    return out


@pytest.mark.parametrize("max_position_embeddings", [262144])
@pytest.mark.parametrize("num_tokens", [1024])
@pytest.mark.parametrize("num_q_heads, num_kv_heads", compute_tp_head_configs())
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("bias", [False])
@torch.inference_mode()
def test_split_rmsnorm_rope(max_position_embeddings, num_tokens, num_q_heads, num_kv_heads,
                                head_size, eps, dtype, seed, bias):
    torch.manual_seed(seed)
    device = DEVICE
    torch.set_default_device(device)

    q_hidden_size = num_q_heads * head_size
    kv_hidden_size = num_kv_heads * head_size
    qkv = torch.randn(num_tokens,
                      q_hidden_size + kv_hidden_size * 2,
                      dtype=dtype,
                      device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)

    if bias:
        q_bias = torch.randn(head_size, dtype=dtype, device=device)
        k_bias = torch.randn(head_size, dtype=dtype, device=device)
    else:
        q_bias = torch.randn(head_size, dtype=dtype, device=device)
        k_bias = torch.randn(head_size, dtype=dtype, device=device)

    sin = torch.randn(num_tokens, head_size, dtype=dtype, device=device)
    cos = torch.randn(num_tokens, head_size, dtype=dtype, device=device)

    q, k, v = triton_split_rmsnorm_rope_impl(
        input=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_size,
        eps=eps,
        sin=sin,
        cos=cos,
        q_bias=q_bias,
        k_bias=k_bias,
        bias=bias,
    )

    q_gold, k_gold, v_gold = split_rmsnorm_rope_ref(
        input=qkv,
        sin=sin,
        cos=cos,
        q_weight=q_weight,
        k_weight=k_weight,
        q_hidden_size=q_hidden_size,
        kv_hidden_size=kv_hidden_size,
        head_dim=head_size,
        eps=eps,
        q_bias=q_bias,
        k_bias=k_bias,
        bias=bias
    )

    torch.testing.assert_close(q.to(torch.float32).cpu(),
                               q_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(k.to(torch.float32).cpu(),
                               k_gold,
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)

    torch.testing.assert_close(v.to(torch.float32).cpu(),
                               v_gold.to(torch.float32),
                               atol=DEFAULT_ATOL,
                               rtol=DEFAULT_RTOL)


if __name__ == "__main__":
    print("Test passed!")

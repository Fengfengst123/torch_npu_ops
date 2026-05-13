import torch
import triton
import triton.language as tl
import pytest

SUPPORTED_V_HEADS_PER_QK = (1, 2, 3, 4)


def get_vectorcore_num() -> int:
    """Return the number of vector cores available on the current NPU device.

    Falls back to a conservative default (20) when the device property is
    not available so the kernel can still be launched in testing environments
    without a real NPU.
    """
    try:
        import torch_npu  # noqa: F401
        props = torch.npu.get_device_properties(torch.npu.current_device())
        # 'vector_core_num' attribute may differ across CANN versions
        return getattr(props, "vector_core_num", 20)
    except Exception:
        return 20


# ---------------------------------------------------------------------------
# Triton kernels (migrated from vllm-ascend PR #6740)
# `v_heads_per_qk` affects the compile-time tile sizes used by tl.arange, so we
# keep one AOT kernel per supported ratio. Other launch parameters still use
# do_not_specialize so each ratio-specific binary can serve many runtime shapes.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["total_rows", "rows_per_vec",
                                "num_heads_qk", "num_heads_v",
                                "qkvz_row_stride", "ba_row_stride",
                                "qkv_row_stride", "z_row_stride",
                                "ba_out_row_stride"])
def fused_qkvzba_split_reshape_cat_gqa_r1_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    # Runtime (do_not_specialize) – will not be folded as constants
    num_heads_qk,
    num_heads_v,
    total_rows,
    rows_per_vec,
    qkvz_row_stride,
    ba_row_stride,
    qkv_row_stride,
    z_row_stride,
    ba_out_row_stride,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
):
    vec_id = tl.program_id(0)

    row_start = vec_id * rows_per_vec
    row_end = tl.minimum(row_start + rows_per_vec, total_rows)

    row_offset = row_start

    # rows_per_iter is always 1; iter_count drives the outer loop.
    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER

    for _ in tl.range(iter_count):
        # rows_per_iter == 1 → tl.arange(0, 1): single-row tile
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end

        # tl.range (runtime loop) because num_heads_qk is do_not_specialize
        for head_id in tl.range(num_heads_qk):
            src_head_offset = head_id * (HEAD_QK * 2 + HEAD_V * 2)

            # Q
            q_range = tl.arange(0, HEAD_QK)
            q_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + q_range[None, :]
            q_dst = row_indices[:, None] * qkv_row_stride + head_id * HEAD_QK + q_range[None, :]
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])

            # K
            k_range = tl.arange(0, HEAD_QK)
            k_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK + k_range[None, :]
            k_dst = row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK + head_id * HEAD_QK + k_range[None, :]
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])

            # V
            v_range = tl.arange(0, HEAD_V)
            v_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + v_range[None, :]
            v_dst = row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK * 2 + head_id * HEAD_V + v_range[None, :]
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])

            # Z
            z_range = tl.arange(0, HEAD_V)
            z_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + HEAD_V + z_range[None, :]
            z_dst = row_indices[:, None] * z_row_stride + head_id * HEAD_V + z_range[None, :]
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])

            # B and A (scalar per v-head, stored interleaved in mixed_ba)
            ba_head_offset = head_id * 2
            b_range = tl.arange(0, 1)
            b_src = row_indices[:, None] * ba_row_stride + ba_head_offset + b_range[None, :]
            b_dst = row_indices[:, None] * ba_out_row_stride + head_id + b_range[None, :]
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])

            # A
            a_src = row_indices[:, None] * ba_row_stride + ba_head_offset + 1 + b_range[None, :]
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])

        row_offset += ROWS_PER_ITER


@triton.jit(do_not_specialize=["total_rows", "rows_per_vec",
                                "num_heads_qk", "num_heads_v",
                                "qkvz_row_stride", "ba_row_stride",
                                "qkv_row_stride", "z_row_stride",
                                "ba_out_row_stride"])
def fused_qkvzba_split_reshape_cat_gqa_r2_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    total_rows,
    rows_per_vec,
    qkvz_row_stride,
    ba_row_stride,
    qkv_row_stride,
    z_row_stride,
    ba_out_row_stride,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
):
    vec_id = tl.program_id(0)

    row_start = vec_id * rows_per_vec
    row_end = tl.minimum(row_start + rows_per_vec, total_rows)

    row_offset = row_start
    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER

    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end

        for head_id in tl.range(num_heads_qk):
            src_head_offset = head_id * (HEAD_QK * 2 + HEAD_V * 4)

            q_range = tl.arange(0, HEAD_QK)
            q_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + q_range[None, :]
            q_dst = row_indices[:, None] * qkv_row_stride + head_id * HEAD_QK + q_range[None, :]
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])

            k_range = tl.arange(0, HEAD_QK)
            k_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK + k_range[None, :]
            k_dst = row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK + head_id * HEAD_QK + k_range[None, :]
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])

            v_range = tl.arange(0, HEAD_V * 2)
            v_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + v_range[None, :]
            v_dst = row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK * 2 + head_id * (HEAD_V * 2) + v_range[None, :]
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])

            z_range = tl.arange(0, HEAD_V * 2)
            z_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + HEAD_V * 2 + z_range[None, :]
            z_dst = row_indices[:, None] * z_row_stride + head_id * (HEAD_V * 2) + z_range[None, :]
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])

            ba_head_offset = head_id * 4
            b_range = tl.arange(0, 2)
            b_src = row_indices[:, None] * ba_row_stride + ba_head_offset + b_range[None, :]
            b_dst = row_indices[:, None] * ba_out_row_stride + head_id * 2 + b_range[None, :]
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])

            a_src = row_indices[:, None] * ba_row_stride + ba_head_offset + 2 + b_range[None, :]
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])

        row_offset += ROWS_PER_ITER


@triton.jit(do_not_specialize=["total_rows", "rows_per_vec",
                                "num_heads_qk", "num_heads_v",
                                "qkvz_row_stride", "ba_row_stride",
                                "qkv_row_stride", "z_row_stride",
                                "ba_out_row_stride"])
def fused_qkvzba_split_reshape_cat_gqa_r3_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    total_rows,
    rows_per_vec,
    qkvz_row_stride,
    ba_row_stride,
    qkv_row_stride,
    z_row_stride,
    ba_out_row_stride,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
):
    vec_id = tl.program_id(0)

    row_start = vec_id * rows_per_vec
    row_end = tl.minimum(row_start + rows_per_vec, total_rows)

    row_offset = row_start
    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER

    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end

        for head_id in tl.range(num_heads_qk):
            src_head_offset = head_id * (HEAD_QK * 2 + HEAD_V * 6)

            q_range = tl.arange(0, HEAD_QK)
            q_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + q_range[None, :]
            q_dst = row_indices[:, None] * qkv_row_stride + head_id * HEAD_QK + q_range[None, :]
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])

            k_range = tl.arange(0, HEAD_QK)
            k_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK + k_range[None, :]
            k_dst = row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK + head_id * HEAD_QK + k_range[None, :]
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])

            v_range = tl.arange(0, HEAD_V * 3)
            v_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + v_range[None, :]
            v_dst = row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK * 2 + head_id * (HEAD_V * 3) + v_range[None, :]
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])

            z_range = tl.arange(0, HEAD_V * 3)
            z_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + HEAD_V * 3 + z_range[None, :]
            z_dst = row_indices[:, None] * z_row_stride + head_id * (HEAD_V * 3) + z_range[None, :]
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])

            ba_head_offset = head_id * 6
            b_range = tl.arange(0, 3)
            b_src = row_indices[:, None] * ba_row_stride + ba_head_offset + b_range[None, :]
            b_dst = row_indices[:, None] * ba_out_row_stride + head_id * 3 + b_range[None, :]
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])

            a_src = row_indices[:, None] * ba_row_stride + ba_head_offset + 3 + b_range[None, :]
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])

        row_offset += ROWS_PER_ITER


@triton.jit(do_not_specialize=["total_rows", "rows_per_vec",
                                "num_heads_qk", "num_heads_v",
                                "qkvz_row_stride", "ba_row_stride",
                                "qkv_row_stride", "z_row_stride",
                                "ba_out_row_stride"])
def fused_qkvzba_split_reshape_cat_gqa_r4_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    total_rows,
    rows_per_vec,
    qkvz_row_stride,
    ba_row_stride,
    qkv_row_stride,
    z_row_stride,
    ba_out_row_stride,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
):
    vec_id = tl.program_id(0)

    row_start = vec_id * rows_per_vec
    row_end = tl.minimum(row_start + rows_per_vec, total_rows)

    row_offset = row_start
    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER

    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end

        for head_id in tl.range(num_heads_qk):
            src_head_offset = head_id * (HEAD_QK * 2 + HEAD_V * 8)

            q_range = tl.arange(0, HEAD_QK)
            q_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + q_range[None, :]
            q_dst = row_indices[:, None] * qkv_row_stride + head_id * HEAD_QK + q_range[None, :]
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])

            k_range = tl.arange(0, HEAD_QK)
            k_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK + k_range[None, :]
            k_dst = row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK + head_id * HEAD_QK + k_range[None, :]
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])

            v_range = tl.arange(0, HEAD_V * 4)
            v_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + v_range[None, :]
            v_dst = row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK * 2 + head_id * (HEAD_V * 4) + v_range[None, :]
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])

            z_range = tl.arange(0, HEAD_V * 4)
            z_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + HEAD_V * 4 + z_range[None, :]
            z_dst = row_indices[:, None] * z_row_stride + head_id * (HEAD_V * 4) + z_range[None, :]
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])

            ba_head_offset = head_id * 8
            b_range = tl.arange(0, 4)
            b_src = row_indices[:, None] * ba_row_stride + ba_head_offset + b_range[None, :]
            b_dst = row_indices[:, None] * ba_out_row_stride + head_id * 4 + b_range[None, :]
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])

            a_src = row_indices[:, None] * ba_row_stride + ba_head_offset + 4 + b_range[None, :]
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])

        row_offset += ROWS_PER_ITER


def get_fused_qkvzba_split_reshape_kernel(v_heads_per_qk: int):
    if v_heads_per_qk == 1:
        return fused_qkvzba_split_reshape_cat_gqa_r1_kernel
    if v_heads_per_qk == 2:
        return fused_qkvzba_split_reshape_cat_gqa_r2_kernel
    if v_heads_per_qk == 3:
        return fused_qkvzba_split_reshape_cat_gqa_r3_kernel
    if v_heads_per_qk == 4:
        return fused_qkvzba_split_reshape_cat_gqa_r4_kernel
    raise ValueError(
        f"Unsupported v_heads_per_qk={v_heads_per_qk}, "
        f"expected one of {SUPPORTED_V_HEADS_PER_QK}"
    )


# ---------------------------------------------------------------------------
# Python-level wrapper (used for JIT-based testing and as the AOT entry point)
# ---------------------------------------------------------------------------

def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    batch = mixed_qkvz.shape[0]
    total_rows = batch  

    v_heads_per_qk = num_heads_v // num_heads_qk
    if v_heads_per_qk not in SUPPORTED_V_HEADS_PER_QK:
        raise ValueError(
            f"Unsupported v_heads_per_qk={v_heads_per_qk}, "
            f"expected one of {SUPPORTED_V_HEADS_PER_QK}"
        )

    v_dim_per_qk = v_heads_per_qk * head_v
    qkvz_dim_t = head_qk * 2 + v_dim_per_qk * 2
    ba_dim_t = v_heads_per_qk * 2

    qkvz_row_stride = num_heads_qk * qkvz_dim_t
    ba_row_stride = num_heads_qk * ba_dim_t
    qkv_row_stride = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    z_row_stride = num_heads_v * head_v
    ba_out_row_stride = num_heads_v

    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [total_rows, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [total_rows, num_heads_v * head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [total_rows, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty(
        [total_rows, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )

    num_vectorcore = get_vectorcore_num()
    grid_size = max(1, min(num_vectorcore, total_rows))
    rows_per_vec = triton.cdiv(total_rows, grid_size)

    # rows_per_iter is do_not_specialize (runtime variable).
    # We fix it to 1 so the binary only stores 1 row per inner tile.
    rows_per_iter = 1

    grid = (grid_size, 1)
    kernel = get_fused_qkvzba_split_reshape_kernel(v_heads_per_qk)
    kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        total_rows,
        rows_per_vec,
        qkvz_row_stride,
        ba_row_stride,
        qkv_row_stride,
        z_row_stride,
        ba_out_row_stride,
        head_qk,
        head_v,
        rows_per_iter,
    )

    # Reshape z from [batch, num_heads_v * head_v] to [batch, num_heads_v, head_v]
    z = z.view(total_rows, num_heads_v, head_v)
    return mixed_qkv, z, b, a


# ---------------------------------------------------------------------------
# Golden / reference implementation (CPU)
# ---------------------------------------------------------------------------

def fused_qkvzba_split_reshape_cat_ref(
    mixed_qkvz: torch.Tensor,
    mixed_ba: torch.Tensor,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
):
    """Pure-PyTorch reference to verify the Triton kernel output.

    Returns (mixed_qkv, z, b, a) with the same shapes as the Triton version.
    """
    batch = mixed_qkvz.shape[0]
    v_heads_per_qk = num_heads_v // num_heads_qk
    qkvz_dim_t = head_qk * 2 + v_heads_per_qk * head_v * 2
    ba_dim_t = v_heads_per_qk * 2

    # Reshape to [batch, num_heads_qk, *_dim_t] for easy slicing
    qkvz = mixed_qkvz.reshape(batch, num_heads_qk, qkvz_dim_t).float()
    ba = mixed_ba.reshape(batch, num_heads_qk, ba_dim_t).float()

    q = qkvz[..., :head_qk]                                          # [B, nqk, head_qk]
    k = qkvz[..., head_qk:2 * head_qk]                               # [B, nqk, head_qk]
    v = qkvz[..., 2 * head_qk:2 * head_qk + v_heads_per_qk * head_v] # [B, nqk, vph*head_v]
    z = qkvz[..., 2 * head_qk + v_heads_per_qk * head_v:]             # [B, nqk, vph*head_v]

    b_ref = ba[..., :v_heads_per_qk]      # [B, nqk, vph]
    a_ref = ba[..., v_heads_per_qk:]      # [B, nqk, vph]

    # Produce flat outputs matching the kernel layout
    mixed_qkv_ref = torch.cat(
        [
            q.reshape(batch, num_heads_qk * head_qk),
            k.reshape(batch, num_heads_qk * head_qk),
            v.reshape(batch, num_heads_v * head_v),
        ],
        dim=-1,
    ).to(mixed_qkvz.dtype)

    z_ref = z.reshape(batch, num_heads_v, head_v).to(mixed_qkvz.dtype)
    b_ref = b_ref.reshape(batch, num_heads_v).to(mixed_ba.dtype)
    a_ref = a_ref.reshape(batch, num_heads_v).to(mixed_ba.dtype)

    return mixed_qkv_ref, z_ref, b_ref, a_ref


# ---------------------------------------------------------------------------
# pytest cases
# Typical Qwen3.5 GatedDeltaNet configurations (after TP split):
#   official checkpoints cover v_heads_per_qk in {1, 2, 3, 4}
#   head_qk = head_v = 128
# ---------------------------------------------------------------------------

# num_heads_qk/num_heads_v are do_not_specialize within each ratio-specific
# kernel, so the cases below compile into four .npubin files: gqa_r1..gqa_r4.
@pytest.mark.parametrize("batch, num_heads_qk, num_heads_v, head_qk, head_v", [
    # (batch, nqk, nv, hqk, hv)
    (1,   8, 16, 128, 128),
    (4,   8, 16, 128, 128),
    (8,   8, 16, 128, 128),
    (1,  16, 16, 128, 128),
    (1,  16, 32, 128, 128),
    (1,  16, 48, 128, 128),
    (1,  16, 64, 128, 128),
    (4,  16, 32, 128, 128),
    (8,  16, 32, 128, 128),
    # Qwen3.5 0.8B / 2B local heads: 16->16 split by TP1/2/4/8.
    (64, 16, 16, 128, 128),
    (64,  8,  8, 128, 128),
    (64,  4,  4, 128, 128),
    (64,  2,  2, 128, 128),
    # Qwen3.5 9B / Qwen3.6 35B-A3B local heads: 16->32 split by TP1/2/4/8.
    (64, 16, 32, 128, 128),
    (64,  8, 16, 128, 128),
    (64,  4,  8, 128, 128),
    (64,  2,  4, 128, 128),
    # Qwen3.5 27B / Qwen3.6 27B local heads: 16->48 split by TP1/2/4/8.
    (64, 16, 48, 128, 128),
    (64,  8, 24, 128, 128),
    (64,  4, 12, 128, 128),
    (64,  2,  6, 128, 128),
    # Qwen3.5 397B-A17B / 122B-A10B local heads: 16->64 split by TP1/2/4/8.
    (64, 16, 64, 128, 128),
    (64,  8, 32, 128, 128),
    (64,  4, 16, 128, 128),
    (64,  2,  8, 128, 128),
    # TP=8
    (4096,  2,  4, 128, 128),
    (4096, 16, 16, 128, 128),
    (4096,  4,  8, 128, 128),
    (4096,  8,  16, 128, 128),
    (4096, 16,  32, 128, 128),
    (4096, 16,  48, 128, 128),
    (4096, 16,  64, 128, 128),
])
def test_fused_qkvzba_split_reshape(
    batch: int,
    num_heads_qk: int,
    num_heads_v: int,
    head_qk: int,
    head_v: int,
    itype=torch.bfloat16,
):
    assert num_heads_v % num_heads_qk == 0, "num_heads_v must be a multiple of num_heads_qk"
    v_heads_per_qk = num_heads_v // num_heads_qk

    qkvz_dim = num_heads_qk * (head_qk * 2 + v_heads_per_qk * head_v * 2)
    ba_dim = num_heads_qk * v_heads_per_qk * 2

    mixed_qkvz_cpu = torch.randn(batch, qkvz_dim, dtype=itype)
    mixed_ba_cpu = torch.randn(batch, ba_dim, dtype=itype)

    # Reference
    ref_qkv, ref_z, ref_b, ref_a = fused_qkvzba_split_reshape_cat_ref(
        mixed_qkvz_cpu, mixed_ba_cpu, num_heads_qk, num_heads_v, head_qk, head_v
    )

    # NPU kernel
    mixed_qkvz_npu = mixed_qkvz_cpu.npu().contiguous()
    mixed_ba_npu = mixed_ba_cpu.npu().contiguous()

    out_qkv, out_z, out_b, out_a = fused_qkvzba_split_reshape_cat(
        mixed_qkvz_npu, mixed_ba_npu, num_heads_qk, num_heads_v, head_qk, head_v
    )

    assert torch.allclose(ref_qkv, out_qkv.cpu(), atol=1e-2, rtol=1e-2), \
        f"mixed_qkv mismatch: max_diff={torch.max(torch.abs(ref_qkv - out_qkv.cpu()))}"
    assert torch.allclose(ref_z, out_z.cpu(), atol=1e-2, rtol=1e-2), \
        f"z mismatch: max_diff={torch.max(torch.abs(ref_z - out_z.cpu()))}"
    assert torch.allclose(ref_b, out_b.cpu(), atol=1e-2, rtol=1e-2), \
        f"b mismatch: max_diff={torch.max(torch.abs(ref_b - out_b.cpu()))}"
    assert torch.allclose(ref_a, out_a.cpu(), atol=1e-2, rtol=1e-2), \
        f"a mismatch: max_diff={torch.max(torch.abs(ref_a - out_a.cpu()))}"

    print(
        f"test_fused_qkvzba_split_reshape PASSED "
        f"batch={batch} nqk={num_heads_qk} nv={num_heads_v} "
        f"hqk={head_qk} hv={head_v}"
    )


if __name__ == "__main__":
    pass

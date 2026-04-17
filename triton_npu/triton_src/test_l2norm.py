import pytest
import torch
import triton
import triton.language as tl


def get_vectorcore_num() -> int:
    try:
        import torch_npu  # noqa: F401
        props = torch.npu.get_device_properties(torch.npu.current_device())
        return getattr(props, "vector_core_num", 20)
    except Exception:
        return 20


@triton.jit(do_not_specialize=["eps", "M", "N", "NUM_ROW_CHUNKS", "NUM_COL_CHUNKS"])
def l2norm_fwd_kernel2_loop(
    X,
    Y,
    eps,
    M,
    N,
    NUM_ROW_CHUNKS,
    NUM_COL_CHUNKS,
    BLOCK_N: tl.constexpr,
    MBLOCK: tl.constexpr,
):
    row_offsets = tl.arange(0, MBLOCK)[:, None]
    col_offsets = tl.arange(0, BLOCK_N)[None, :]
    base_row = tl.program_id(0) * (NUM_ROW_CHUNKS * MBLOCK)

    for row_chunk in tl.range(0, NUM_ROW_CHUNKS):
        row_idx = base_row + row_chunk * MBLOCK + row_offsets
        row_mask = row_idx < M
        square_sum = tl.zeros((MBLOCK, 1), dtype=tl.float32)

        for col_chunk in tl.range(0, NUM_COL_CHUNKS):
            col_idx = col_chunk * BLOCK_N + col_offsets
            mask = row_mask & (col_idx < N)
            xs = tl.load(X + row_idx * N + col_idx, mask=mask, other=0.0).to(
                tl.float32
            )
            square_sum += tl.sum(xs * xs, 1)[:, None]

        rsqrt = tl.rsqrt(square_sum + eps)
        for col_chunk in tl.range(0, NUM_COL_CHUNKS):
            col_idx = col_chunk * BLOCK_N + col_offsets
            mask = row_mask & (col_idx < N)
            xs = tl.load(X + row_idx * N + col_idx, mask=mask, other=0.0).to(
                tl.float32
            )
            tl.store(Y + row_idx * N + col_idx, xs * rsqrt, mask=mask)


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1]).contiguous()
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)

    assert x.dtype == torch.bfloat16
    assert x.device.type == "npu"
    assert x.stride(-1) == 1

    rows, cols = x.shape
    assert cols > 0, "hidden dim must be positive"

    block_n = 128
    mblock = 69
    num_core = get_vectorcore_num()
    main_bs = triton.cdiv(rows, num_core)
    num_row_chunks = triton.cdiv(main_bs, mblock)
    num_col_chunks = triton.cdiv(cols, block_n)
    grid = (num_core,)
    l2norm_fwd_kernel2_loop[grid](
        X=x,
        Y=y,
        eps=eps,
        M=rows,
        N=cols,
        NUM_ROW_CHUNKS=num_row_chunks,
        NUM_COL_CHUNKS=num_col_chunks,
        BLOCK_N=block_n,
        MBLOCK=mblock,
    )
    return y.view(x_shape_og)


def l2norm_ref(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    out = x_fp32 * torch.rsqrt(torch.sum(x_fp32 * x_fp32, dim=-1, keepdim=True) + eps)
    return out.to(x.dtype)


@pytest.mark.parametrize(
    "shape",
    [(1, 31, 4, 64), (1, 31, 4, 128), (2, 17, 3, 192), (2, 64, 8, 256)],
)
def test_l2norm_fwd(shape):
    if not torch.npu.is_available():
        pytest.skip("NPU device not available")

    x_cpu = torch.randn(shape, dtype=torch.bfloat16)
    ref = l2norm_ref(x_cpu)

    x_npu = x_cpu.npu()
    out_npu = l2norm_fwd(x_npu)

    assert torch.allclose(out_npu.cpu(), ref, atol=5e-2, rtol=5e-2)

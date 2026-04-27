import torch
import triton
import torch_npu
import triton.language as tl
import torch.nn.functional as F
from typing import Optional, Tuple
import pytest
import ctypes


# =============================================================================
# Fast Kernel (optimized for group_size<=128, grid=physical core count)
# =============================================================================
# BLOCK_N is fixed to 128 as tl.constexpr.
# N_CORES is a runtime param (do_not_specialize) to avoid multiple binaries.
# HAS_BIAS/HAS_Z remain heuristics so that the compiler can optimize branches.
# =============================================================================
@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
})
@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    # Continuous row assignment for better memory locality
    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        B_base = B + group * N
        b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        if HAS_Z:
            Z_base = Z + curr_row * stride_z_row + group * N
        if not IS_RMS_NORM:
            Mean_base = Mean + curr_row + group * M
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=0) / N
            tl.store(Mean_base, mean)
            xbar = tl.where(mask, x - mean, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        else:
            xbar = tl.where(mask, x, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd_base, rstd)

        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y, mask=mask)


@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
})
@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    # Continuous row assignment for better memory locality
    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        B_base = B + group * N
        b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        if HAS_Z:
            Z_base = Z + curr_row * stride_z_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y, mask=mask)


# =============================================================================
# Fast Kernel (BF16 LayerNorm variant)
# =============================================================================
@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
})
@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_bf16(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        B_base = B + group * N
        b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        if HAS_Z:
            Z_base = Z + curr_row * stride_z_row + group * N
        Mean_base = Mean + curr_row + group * M
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean_base, mean)
        xbar = tl.where(mask, x - mean, 0.)
        var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd_base, rstd)

        x_hat = (x - mean) * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y.to(tl.bfloat16), mask=mask)


# =============================================================================
# Fast Kernel (BF16 RMSNorm variant)
# =============================================================================
@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
})
@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_bf16(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        B_base = B + group * N
        b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        if HAS_Z:
            Z_base = Z + curr_row * stride_z_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y.to(tl.bfloat16), mask=mask)


# =============================================================================
# Fast Kernel with Z (HAS_Z=True variants)
# =============================================================================
@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
})
@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_z(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    HAS_BIAS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        B_base = B + group * N
        b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Z_base = Z + curr_row * stride_z_row + group * N
        if not IS_RMS_NORM:
            Mean_base = Mean + curr_row + group * M
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=0) / N
            tl.store(Mean_base, mean)
            xbar = tl.where(mask, x - mean, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        else:
            xbar = tl.where(mask, x, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd_base, rstd)

        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y, mask=mask)


@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
})
@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_z(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    HAS_BIAS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        B_base = B + group * N
        b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Z_base = Z + curr_row * stride_z_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y, mask=mask)


@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
})
@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_bf16_z(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    HAS_BIAS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        B_base = B + group * N
        b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Z_base = Z + curr_row * stride_z_row + group * N
        Mean_base = Mean + curr_row + group * M
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean_base, mean)
        xbar = tl.where(mask, x - mean, 0.)
        var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd_base, rstd)

        x_hat = (x - mean) * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y.to(tl.bfloat16), mask=mask)


@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
})
@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_bf16_z(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    HAS_BIAS: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        B_base = B + group * N
        b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Z_base = Z + curr_row * stride_z_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y.to(tl.bfloat16), mask=mask)


# =============================================================================
# RMS fast kernels split by HAS_BIAS to produce distinct AOT binaries
# =============================================================================
@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_bias(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    B_base = B + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w + b
        tl.store(Y_base + cols, y, mask=mask)


@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_nobias(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w
        tl.store(Y_base + cols, y, mask=mask)


@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_bf16_bias(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    B_base = B + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w + b
        tl.store(Y_base + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_bf16_nobias(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w
        tl.store(Y_base + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_z_bias(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    B_base = B + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Z_base = Z + curr_row * stride_z_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w + b
        if NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y, mask=mask)


@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_z_nobias(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Z_base = Z + curr_row * stride_z_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w
        if NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y, mask=mask)


@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_bf16_z_bias(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    B_base = B + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Z_base = Z + curr_row * stride_z_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w + b
        if NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit(
    do_not_specialize=["M", "N", "eps", "N_CORES"]
)
def layer_norm_fwd_kernel_fast_rms_bf16_z_nobias(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    N_CORES,
    NORM_BEFORE_GATE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    if row < remain:
        start_row = row * n_iters
    else:
        start_row = remain * (base_iters + 1) + (row - remain) * base_iters

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = start_row + i
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        Z_base = Z + curr_row * stride_z_row + group * N
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        mean_sq = tl.sum(x * x, axis=0) / N
        rstd = 1 / tl.sqrt(mean_sq + eps)
        tl.store(Rstd_base, rstd)

        x_hat = x * rstd
        y = x_hat * w
        if NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y.to(tl.bfloat16), mask=mask)


# =============================================================================
# Fallback Kernel (original implementation, grid=MAX_CORES)
# =============================================================================
@triton.heuristics({
    "HAS_BIAS": lambda args: args["B"] is not None,
    "HAS_Z": lambda args: args["Z"] is not None,
})
@triton.jit(do_not_specialize=["M", "N", "eps"])
def layer_norm_fwd_kernel(
    X, Y, W, B, Z, Mean, Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    N_CORES: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    BLOCK_ROWS = M if M < N_CORES else N_CORES
    base_iters = M // BLOCK_ROWS
    remain = M % BLOCK_ROWS
    n_iters = base_iters
    if row < remain:
        n_iters = n_iters + 1

    # Interleaved row assignment (original logic)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    W_base = W + group * N
    w = tl.load(W_base + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        B_base = B + group * N
        b = tl.load(B_base + cols, mask=mask).to(tl.float32)

    for i in tl.range(n_iters):
        curr_row = i * BLOCK_ROWS + row
        X_base = X + curr_row * stride_x_row + group * N
        Y_base = Y + curr_row * stride_y_row + group * N
        if HAS_Z:
            Z_base = Z + curr_row * stride_z_row + group * N
        if not IS_RMS_NORM:
            Mean_base = Mean + curr_row + group * M
        Rstd_base = Rstd + curr_row + group * M

        x = tl.load(X_base + cols, mask=mask, other=0.).to(tl.float32)
        if HAS_Z and not NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            x *= z * tl.sigmoid(z)
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=0) / N
            tl.store(Mean_base, mean)
            xbar = tl.where(mask, x - mean, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        else:
            xbar = tl.where(mask, x, 0.)
            var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd_base, rstd)

        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        if HAS_Z and NORM_BEFORE_GATE:
            z = tl.load(Z_base + cols, mask=mask).to(tl.float32)
            y *= z * tl.sigmoid(z)
        tl.store(Y_base + cols, y, mask=mask)


# =============================================================================
# Python wrappers
# =============================================================================
def _layer_norm_fwd_fast(
    x, weight, bias, eps,
    z=None, group_size=None,
    norm_before_gate=True, is_rms_norm=False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1

    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N, )
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N, )

    out = torch.empty_like(x)
    mean = (torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
            if not is_rms_norm else None)
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    if group_size > 128:
        raise RuntimeError(
            "Fast layer_norm kernel only supports feature dim <= 128.")

    try:
        rt_lib = ctypes.CDLL('/usr/local/Ascend/ascend-toolkit/latest/lib64/libruntime.so')
        ai_core_cnt = ctypes.c_uint32()
        ret = rt_lib.rtGetAiCoreCount(ctypes.byref(ai_core_cnt))
        vector_cores = ai_core_cnt.value * 2 if ret == 0 else 40
    except Exception:
        vector_cores = 40

    n_cores = min(M, vector_cores)
    grid = (n_cores, ngroups)

    with torch.npu.device(x.device.index):
        if x.dtype == torch.bfloat16:
            if is_rms_norm:
                if bias is None:
                    layer_norm_fwd_kernel_fast_rms_bf16_nobias[grid](
                        x, out, weight, bias, z, mean, rstd,
                        x.stride(0), out.stride(0),
                        z.stride(0) if z is not None else 0,
                        M, group_size, eps, n_cores,
                        NORM_BEFORE_GATE=norm_before_gate,
                        BLOCK_N=128,
                    )
                else:
                    layer_norm_fwd_kernel_fast_rms_bf16_bias[grid](
                        x, out, weight, bias, z, mean, rstd,
                        x.stride(0), out.stride(0),
                        z.stride(0) if z is not None else 0,
                        M, group_size, eps, n_cores,
                        NORM_BEFORE_GATE=norm_before_gate,
                        BLOCK_N=128,
                    )
            else:
                layer_norm_fwd_kernel_fast_bf16[grid](
                    x, out, weight, bias, z, mean, rstd,
                    x.stride(0), out.stride(0),
                    z.stride(0) if z is not None else 0,
                    M, group_size, eps, n_cores,
                    NORM_BEFORE_GATE=norm_before_gate,
                    BLOCK_N=128,
                )
        else:
            if is_rms_norm:
                if bias is None:
                    layer_norm_fwd_kernel_fast_rms_nobias[grid](
                        x, out, weight, bias, z, mean, rstd,
                        x.stride(0), out.stride(0),
                        z.stride(0) if z is not None else 0,
                        M, group_size, eps, n_cores,
                        NORM_BEFORE_GATE=norm_before_gate,
                        BLOCK_N=128,
                    )
                else:
                    layer_norm_fwd_kernel_fast_rms_bias[grid](
                        x, out, weight, bias, z, mean, rstd,
                        x.stride(0), out.stride(0),
                        z.stride(0) if z is not None else 0,
                        M, group_size, eps, n_cores,
                        NORM_BEFORE_GATE=norm_before_gate,
                        BLOCK_N=128,
                    )
            else:
                layer_norm_fwd_kernel_fast[grid](
                    x, out, weight, bias, z, mean, rstd,
                    x.stride(0), out.stride(0),
                    z.stride(0) if z is not None else 0,
                    M, group_size, eps, n_cores,
                    NORM_BEFORE_GATE=norm_before_gate,
                    IS_RMS_NORM=is_rms_norm,
                    BLOCK_N=128,
                )
    return out, mean, rstd


def _layer_norm_fwd_fast_z(
    x, weight, bias, eps,
    z=None, group_size=None,
    norm_before_gate=True, is_rms_norm=False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1

    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N, )
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N, )

    out = torch.empty_like(x)
    mean = (torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
            if not is_rms_norm else None)
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    if group_size > 128:
        raise RuntimeError(
            "Fast layer_norm kernel only supports feature dim <= 128.")

    try:
        rt_lib = ctypes.CDLL('/usr/local/Ascend/ascend-toolkit/latest/lib64/libruntime.so')
        ai_core_cnt = ctypes.c_uint32()
        ret = rt_lib.rtGetAiCoreCount(ctypes.byref(ai_core_cnt))
        vector_cores = ai_core_cnt.value * 2 if ret == 0 else 40
    except Exception:
        vector_cores = 40

    n_cores = min(M, vector_cores)
    grid = (n_cores, ngroups)

    with torch.npu.device(x.device.index):
        if x.dtype == torch.bfloat16:
            if is_rms_norm:
                if bias is None:
                    layer_norm_fwd_kernel_fast_rms_bf16_z_nobias[grid](
                        x, out, weight, bias, z, mean, rstd,
                        x.stride(0), out.stride(0),
                        z.stride(0) if z is not None else 0,
                        M, group_size, eps, n_cores,
                        NORM_BEFORE_GATE=norm_before_gate,
                        BLOCK_N=128,
                    )
                else:
                    layer_norm_fwd_kernel_fast_rms_bf16_z_bias[grid](
                        x, out, weight, bias, z, mean, rstd,
                        x.stride(0), out.stride(0),
                        z.stride(0) if z is not None else 0,
                        M, group_size, eps, n_cores,
                        NORM_BEFORE_GATE=norm_before_gate,
                        BLOCK_N=128,
                    )
            else:
                layer_norm_fwd_kernel_fast_bf16_z[grid](
                    x, out, weight, bias, z, mean, rstd,
                    x.stride(0), out.stride(0),
                    z.stride(0) if z is not None else 0,
                    M, group_size, eps, n_cores,
                    NORM_BEFORE_GATE=norm_before_gate,
                    BLOCK_N=128,
                )
        else:
            if is_rms_norm:
                if bias is None:
                    layer_norm_fwd_kernel_fast_rms_z_nobias[grid](
                        x, out, weight, bias, z, mean, rstd,
                        x.stride(0), out.stride(0),
                        z.stride(0) if z is not None else 0,
                        M, group_size, eps, n_cores,
                        NORM_BEFORE_GATE=norm_before_gate,
                        BLOCK_N=128,
                    )
                else:
                    layer_norm_fwd_kernel_fast_rms_z_bias[grid](
                        x, out, weight, bias, z, mean, rstd,
                        x.stride(0), out.stride(0),
                        z.stride(0) if z is not None else 0,
                        M, group_size, eps, n_cores,
                        NORM_BEFORE_GATE=norm_before_gate,
                        BLOCK_N=128,
                    )
            else:
                layer_norm_fwd_kernel_fast_z[grid](
                    x, out, weight, bias, z, mean, rstd,
                    x.stride(0), out.stride(0),
                    z.stride(0) if z is not None else 0,
                    M, group_size, eps, n_cores,
                    NORM_BEFORE_GATE=norm_before_gate,
                    IS_RMS_NORM=is_rms_norm,
                    BLOCK_N=128,
                )
    return out, mean, rstd


def _layer_norm_fwd_fallback(
    x, weight, bias, eps,
    z=None, group_size=None,
    norm_before_gate=True, is_rms_norm=False,
):
    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1

    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N, )
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N, )

    out = torch.empty_like(x)
    mean = (torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
            if not is_rms_norm else None)
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")

    n_cores = min(M, 65535)
    grid = (n_cores, ngroups)

    with torch.npu.device(x.device.index):
        layer_norm_fwd_kernel[grid](
            x, out, weight, bias, z, mean, rstd,
            x.stride(0), out.stride(0),
            z.stride(0) if z is not None else 0,
            M, group_size, eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            N_CORES=n_cores,
        )
    return out, mean, rstd


def layer_norm_fwd(
    x, weight, bias=None, eps=1e-6,
    z=None, group_size=None,
    norm_before_gate=True, is_rms_norm=False,
):
    """Unified entry: auto-select fast or fallback kernel."""
    M, N = x.shape
    if group_size is None:
        group_size = N

    if group_size <= 128:
        if z is not None:
            out, mean, rstd = _layer_norm_fwd_fast_z(
                x, weight, bias, eps, z,
                group_size=group_size,
                norm_before_gate=norm_before_gate,
                is_rms_norm=is_rms_norm,
            )
        else:
            out, mean, rstd = _layer_norm_fwd_fast(
                x, weight, bias, eps, z,
                group_size=group_size,
                norm_before_gate=norm_before_gate,
                is_rms_norm=is_rms_norm,
            )
    else:
        out, mean, rstd = _layer_norm_fwd_fallback(
            x, weight, bias, eps, z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
    return out


# =============================================================================
# Pytest validation
# =============================================================================
def _golden_ref(x, weight, bias, eps, z=None, group_size=None,
                norm_before_gate=True, is_rms_norm=False):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1]).cpu().contiguous()
    M, N = x.shape
    weight = weight.cpu().contiguous()
    bias = bias.cpu().contiguous() if bias is not None else None
    z = z.reshape(-1, z.shape[-1]).cpu().contiguous() if z is not None else None
    if group_size is None:
        group_size = N
    ngroups = N // group_size

    if z is not None and not norm_before_gate:
        gate = z * torch.sigmoid(z)
        x = x * gate

    x_grouped = x.unfold(dimension=1, size=group_size, step=group_size)
    x_grouped = x_grouped.reshape(-1, group_size)

    if not is_rms_norm:
        x_norm = torch.layer_norm(
            x_grouped, normalized_shape=(group_size,),
            weight=None, bias=None, eps=eps
        )
    else:
        mean_sq = torch.mean(x_grouped ** 2, dim=-1, keepdim=True)
        x_norm = x_grouped * torch.rsqrt(mean_sq + eps)

    x_norm = x_norm.reshape(M, ngroups, group_size)
    x_norm = x_norm.contiguous().view(M, N)
    y = x_norm * weight
    if bias is not None:
        y = y + bias
    if z is not None and norm_before_gate:
        gate = z * torch.sigmoid(z)
        y = y * gate
    return y.reshape(x_shape_og)


@pytest.mark.parametrize(
    "batch,seq,feat,bias,z,gate,rms,gs,dtype",
    [
        (2, 8, 128, True, False, True, False, None, torch.float32),
        (32768, 1, 128, True, False, True, False, None, torch.float32),
        (2, 8, 128, False, False, True, True, None, torch.float32),
        (2, 8, 128, True, False, True, True, None, torch.float32),
        (32768, 1, 128, False, False, True, True, None, torch.float32),
        (2, 8, 128, True, False, True, False, None, torch.bfloat16),
        (2, 8, 128, False, False, True, True, None, torch.bfloat16),
        (2, 8, 128, True, False, True, True, None, torch.bfloat16),
        # z non-empty fast path tests
        (2, 8, 128, True, True, True, False, None, torch.float32),
        (2, 8, 128, False, True, True, True, None, torch.float32),
        (2, 8, 128, True, True, True, True, None, torch.float32),
        (112, 1, 128, False, True, True, True, 128, torch.float32),
        (2, 8, 128, True, True, True, False, None, torch.bfloat16),
        (2, 8, 128, False, True, True, True, None, torch.bfloat16),
        (2, 8, 128, True, True, True, True, None, torch.bfloat16),
    ],
)
def test_layer_norm_fast(batch, seq, feat, bias, z, gate, rms, gs, dtype):
    torch.manual_seed(42)
    x = torch.randn(batch, seq, feat, dtype=dtype, device="npu")
    w = torch.randn(feat, dtype=dtype, device="npu")
    b = torch.randn(feat, dtype=dtype, device="npu") if bias else None
    z_t = torch.randn(batch, seq, feat, dtype=dtype, device="npu") if z else None

    out = layer_norm_fwd(
        x.reshape(-1, feat), w, b, eps=1e-6,
        z=z_t.reshape(-1, feat) if z else None,
        group_size=gs if gs else feat,
        norm_before_gate=gate, is_rms_norm=rms,
    )

    ref = _golden_ref(x, w, b, eps=1e-6, z=z_t,
                      group_size=gs if gs else feat,
                      norm_before_gate=gate, is_rms_norm=rms)

    max_abs = torch.max(torch.abs(out.cpu().reshape(ref.shape).to(torch.float32) - ref.to(torch.float32))).item()
    # bf16 has lower precision, so use a looser tolerance
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-3
    assert max_abs < tol, f"abs err {max_abs} > {tol}"

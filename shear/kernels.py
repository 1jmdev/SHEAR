import torch
import triton
import triton.language as tl


@triton.jit
def swiglu_weight_kernel(gate_up, weights, output, rows, CHANNELS: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    column = tl.arange(0, CHANNELS)
    valid = (row < rows)[:, None]
    base = row.to(tl.int64)[:, None] * 2 * CHANNELS + column[None, :]
    gate = tl.load(gate_up + base, mask=valid).to(tl.float32)
    up = tl.load(gate_up + base + CHANNELS, mask=valid).to(tl.float32)
    weight = tl.load(weights + row, mask=row < rows).to(tl.float32)[:, None]
    result = weight * gate * tl.sigmoid(gate) * up
    tl.store(output + row.to(tl.int64)[:, None] * CHANNELS + column[None, :], result.to(output.dtype.element_ty), mask=valid)


@triton.jit
def segmented_gram_kernel(
    left, right, offsets, output, CHANNELS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    expert = tl.program_id(0)
    columns_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns_n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    start = tl.load(offsets + expert)
    end = tl.load(offsets + expert + 1)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for block in range(start, end, BLOCK_K):
        row = block + tl.arange(0, BLOCK_K)
        valid = (row < end)[:, None]
        row = row.to(tl.int64)[:, None] * CHANNELS
        first = tl.load(left + row + columns_m[None, :], mask=valid, other=0.0)
        second = tl.load(right + row + columns_n[None, :], mask=valid, other=0.0)
        accumulator += tl.dot(tl.trans(first), second)
    pointer = output + expert.to(tl.int64) * CHANNELS * CHANNELS + columns_m[:, None] * CHANNELS + columns_n[None, :]
    tl.store(pointer, tl.load(pointer) + accumulator)


def swiglu_weight(gate_up, weights):
    rows, channels = gate_up.shape[0], gate_up.shape[1] // 2
    output = torch.empty(rows, channels, device=gate_up.device, dtype=gate_up.dtype)
    swiglu_weight_kernel[(triton.cdiv(rows, 16),)](gate_up, weights, output, rows, channels, 16)
    return output


def accumulate_segmented_gram(left, right, offsets, output):
    experts, channels = output.shape[0], output.shape[1]
    block = min(64, channels)
    grid = (experts, channels // block, channels // block)
    segmented_gram_kernel[grid](left, right, offsets, output, channels, block, block, 32)

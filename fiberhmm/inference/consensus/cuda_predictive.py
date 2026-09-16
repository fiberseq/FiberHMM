"""Optional fused CUDA reduction; imported only by the explicit CUDA backend."""
import torch
import triton
import triton.language as tl


@triton.jit
def _loss_kernel(prefix, starts, ends, penalty, output, n_rows,
                 width, geometries,
                 ROWS: tl.constexpr, COLS: tl.constexpr):
    row = (tl.program_id(0)*ROWS+tl.arange(0, ROWS)).to(tl.int64)
    column = tl.arange(0, COLS)
    best = tl.full((ROWS,), -float('inf'), tl.float64)
    explained = tl.full((ROWS,), -float('inf'), tl.float64)
    for begin in range(tl.cdiv(geometries, COLS)):
        g = begin*COLS+column
        a = tl.load(starts+g, g < geometries, other=0)
        b = tl.load(ends+g, g < geometries, other=0)
        d = tl.load(penalty+g, g < geometries, other=-float('inf'))
        left = tl.load(prefix+row[:, None]*width+a[None, :],
                       (row[:, None] < n_rows) & (g[None, :] < geometries), other=0.)
        right = tl.load(prefix+row[:, None]*width+b[None, :],
                        (row[:, None] < n_rows) & (g[None, :] < geometries), other=0.)
        value = tl.where(g[None, :] < geometries, right-left, -float('inf'))
        best = tl.maximum(best, tl.max(value, axis=1))
        explained = tl.maximum(explained, tl.max(value+d[None, :], axis=1))
    tl.store(output+row, best-explained, row < n_rows)


def profile_losses(prefixes, starts, ends, penalties):
    result = torch.empty((len(prefixes),), device=prefixes.device, dtype=torch.float64)
    _loss_kernel[(triton.cdiv(len(prefixes), 4),)](
        prefixes, starts, ends, penalties, result, len(prefixes), prefixes.shape[1],
        len(starts), 4, 128, enable_fp_fusion=False)
    return result

import math

import pytest
import torch

from flash_mla import (
    flash_mla_dense_decode_sm100,
    flash_mla_dense_decode_sm100_out,
    get_sm100_dense_decode_workspace_size,
)


if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
    pytest.skip("SM100 dense MLA decode requires compute capability 10.0", allow_module_level=True)


def _reference(q_nope, q_pe, kv_cache, seq_lens, block_table, scale):
    outputs = []
    lses = []
    for batch_idx in range(q_nope.shape[0]):
        cache = kv_cache[block_table[batch_idx]].flatten(0, 1)[: int(seq_lens[batch_idx])]
        scores = (
            q_nope[batch_idx].float() @ cache[:, :512].float().T
            + q_pe[batch_idx].float() @ cache[:, 512:].float().T
        ) * scale
        outputs.append((scores.softmax(-1) @ cache[:, :512].float()).to(q_nope.dtype))
        lses.append(torch.logsumexp(scores, dim=-1))
    return torch.stack(outputs), torch.stack(lses)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("page_size", [64, 128])
def test_dense_decode_sm100_matches_reference(dtype, page_size):
    torch.manual_seed(7)
    batch_size, num_heads = 2, 20
    seq_lens = torch.tensor([191, 257], device="cuda", dtype=torch.int32)
    table_width = math.ceil(int(seq_lens.max()) / page_size)
    pack_factor = 128 // page_size
    table_width = math.ceil(table_width / pack_factor) * pack_factor
    block_table = torch.arange(
        batch_size * table_width, device="cuda", dtype=torch.int32
    ).view(batch_size, table_width)
    kv_cache = torch.randn(
        batch_size * table_width, page_size, 576, device="cuda", dtype=dtype
    )
    q_nope = torch.randn(batch_size, num_heads, 512, device="cuda", dtype=dtype)
    q_pe = torch.randn(batch_size, num_heads, 64, device="cuda", dtype=dtype)
    scale = 1 / 16

    actual = flash_mla_dense_decode_sm100(
        q_nope, q_pe, kv_cache, seq_lens, block_table, scale
    )
    expected, _ = _reference(
        q_nope, q_pe, kv_cache, seq_lens, block_table, scale
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_dense_decode_sm100_out_captures_with_lse():
    torch.manual_seed(11)
    batch_size, num_heads, page_size = 2, 20, 128
    seq_lens = torch.tensor([191, 257], device="cuda", dtype=torch.int32)
    table_width = math.ceil(int(seq_lens.max()) / page_size)
    block_table = torch.arange(
        batch_size * table_width, device="cuda", dtype=torch.int32
    ).view(batch_size, table_width)
    kv_cache = torch.randn(
        batch_size * table_width,
        page_size,
        576,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_nope = torch.zeros(
        batch_size, 128, 512, device="cuda", dtype=torch.bfloat16
    )
    q_pe = torch.zeros(
        batch_size, 128, 64, device="cuda", dtype=torch.bfloat16
    )
    q_nope[:, :num_heads].normal_()
    q_pe[:, :num_heads].normal_()
    out = torch.empty_like(q_nope)
    lse = torch.empty(batch_size, 128, device="cuda", dtype=torch.float32)
    workspace_size = get_sm100_dense_decode_workspace_size(
        table_width * page_size, batch_size
    )
    workspace = torch.empty(workspace_size, device="cuda", dtype=torch.uint8)

    def run():
        flash_mla_dense_decode_sm100_out(
            q_nope,
            q_pe,
            kv_cache,
            seq_lens,
            block_table,
            workspace,
            1 / 16,
            out=out,
            lse=lse,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()

    expected, expected_lse = _reference(
        q_nope[:, :num_heads],
        q_pe[:, :num_heads],
        kv_cache,
        seq_lens,
        block_table,
        1 / 16,
    )
    torch.testing.assert_close(
        out[:, :num_heads], expected, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        lse[:, :num_heads], expected_lse, atol=2e-2, rtol=2e-2
    )

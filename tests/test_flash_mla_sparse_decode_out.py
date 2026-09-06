import lib
import pytest
import torch
from lib import RawTestParamForDecode

import flash_mla


@torch.inference_mode()
@pytest.mark.parametrize(("num_heads", "query_length"), [(64, 1), (128, 3)])
def test_sparse_decode_out_matches_allocating_api_and_replays(num_heads, query_length):
    """Caller-owned output/workspace must preserve results under CUDA graph replay."""

    assert torch.cuda.is_available()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            params = RawTestParamForDecode(
                b=4,
                h_q=num_heads,
                s_q=query_length,
                h_kv=1,
                s_kv=512,
                is_varlen=True,
                topk=64,
                block_size=256,
                d_qk=576,
                have_topk_length=True,
                enable_attn_sink=True,
                num_runs=0,
                seed=123,
            ).to_test_param()
            testcase = lib.generate_testcase_for_decode(params)

        reference_scheduler, _ = flash_mla.get_mla_metadata()
        expected_out, expected_lse = lib.run_flash_mla_decode(
            params, testcase, reference_scheduler, None
        )
        scheduler, _ = flash_mla.get_mla_metadata()
        assert not scheduler.have_initialized
        q = testcase.q
        kv = testcase.kv_scope.get_kvcache_for_flash_mla()
        indices = testcase.kv_scope.indices_in_kvcache
        workspace_bytes = flash_mla.get_sparse_decode_workspace_size(
            q.shape[0], q.shape[1], q.shape[2], q.shape[3], params.d_v
        )
        out = torch.empty(
            *q.shape[:3], params.d_v, dtype=torch.bfloat16, device=q.device
        )
        lse = torch.empty(*q.shape[:3], dtype=torch.float32, device=q.device)
        # Pass an aligned interior view and retain canaries on both sides. This
        # checks that the public size query and the C++ workspace partitioning
        # agree exactly, rather than merely allocating a generous buffer.
        guard_bytes = 256
        workspace_storage = torch.full(
            (workspace_bytes + 2 * guard_bytes,),
            0xA5,
            dtype=torch.uint8,
            device=q.device,
        )
        workspace = workspace_storage[guard_bytes:-guard_bytes]

        def run_out():
            flash_mla.flash_mla_sparse_decode_fwd_out(
                q,
                kv,
                indices,
                scheduler,
                out=out,
                lse=lse,
                workspace=workspace,
                head_dim_v=params.d_v,
                softmax_scale=testcase.sm_scale,
                topk_length=testcase.kv_scope.topk_length,
                attn_sink=testcase.attn_sink,
            )

        # Scheduler metadata is the only persistent allocation permitted on
        # the first eager call. In particular, initialization must not create
        # the old split-KV output/accumulator tensors transiently: at Harmony's
        # largest graph bucket those were roughly a gigabyte.
        torch.cuda.reset_peak_memory_stats(q.device)
        first_allocated_before = torch.cuda.memory_allocated(q.device)
        run_out()
        torch.cuda.synchronize()
        first_allocated_after = torch.cuda.memory_allocated(q.device)
        first_peak = torch.cuda.max_memory_allocated(q.device)
        assert (
            first_peak - max(first_allocated_before, first_allocated_after)
            < 1024 * 1024
        )
        assert scheduler.have_initialized
        torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(lse.transpose(1, 2), expected_lse, rtol=0, atol=0)
        assert torch.all(workspace_storage[:guard_bytes] == 0xA5)
        assert torch.all(workspace_storage[-guard_bytes:] == 0xA5)

        # The first out call may load cubins, but a warmed call must not retain
        # any PyTorch-managed device allocation. This is the property Harmony
        # relies on when 100+ Rust graph variants share one planned arena.
        torch.cuda.reset_peak_memory_stats(q.device)
        allocated_before = torch.cuda.memory_allocated(q.device)
        run_out()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated(q.device) == allocated_before
        assert torch.cuda.max_memory_allocated(q.device) == allocated_before

        with pytest.raises(
            RuntimeError, match="workspace has .* bytes, but .* are required"
        ):
            flash_mla.flash_mla_sparse_decode_fwd_out(
                q,
                kv,
                indices,
                scheduler,
                out=out,
                lse=lse,
                workspace=workspace[:-1],
                head_dim_v=params.d_v,
                softmax_scale=testcase.sm_scale,
                topk_length=testcase.kv_scope.topk_length,
                attn_sink=testcase.attn_sink,
            )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_out()
        next_q = torch.randn_like(q).clamp_(-1, 1)
        q.copy_(next_q)
        graph.replay()
        torch.cuda.synchronize()
        replay_out = out.clone()
        replay_lse = lse.clone()
        eager_out, eager_lse = lib.run_flash_mla_decode(
            params, testcase, reference_scheduler, None
        )
        torch.testing.assert_close(replay_out, eager_out, rtol=0, atol=0)
        torch.testing.assert_close(
            replay_lse.transpose(1, 2), eager_lse, rtol=0, atol=0
        )
        assert torch.all(workspace_storage[:guard_bytes] == 0xA5)
        assert torch.all(workspace_storage[-guard_bytes:] == 0xA5)
    finally:
        torch.set_default_dtype(previous_dtype)

#pragma once

#include <torch/extension.h>

void dense_decode_sm100_fwd_out(
    const at::Tensor& out,
    const at::Tensor& lse,
    const at::Tensor& q_nope,
    const at::Tensor& q_pe,
    const at::Tensor& kv_cache,
    const at::Tensor& cache_seqlens,
    const at::Tensor& block_table,
    const at::Tensor& workspace,
    double softmax_scale,
    int64_t num_kv_splits);

int64_t dense_decode_sm100_workspace_size(
    int64_t max_seqlen,
    int64_t batch_size,
    int64_t sm_count,
    int64_t num_kv_splits);

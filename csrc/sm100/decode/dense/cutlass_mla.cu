/*
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "api/dense_decode_sm100.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <iostream>

#include "device/sm100_mla.hpp"
#include "kernel/sm100_mla_tile_scheduler.hpp"

// clang-format off
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12040
void dense_decode_sm100_fwd_out(
    torch::Tensor const& out,
    torch::Tensor const& lse,
    torch::Tensor const& q_nope,
    torch::Tensor const& q_pe,
    torch::Tensor const& kv_c_and_k_pe_cache,
    torch::Tensor const& seq_lens,
    torch::Tensor const& page_table,
    torch::Tensor const& workspace,
    double sm_scale,
    int64_t num_kv_splits) {
  (void)sm_scale;
  TORCH_CHECK(false, "CUDA version must be >= 12.4 for dense_decode_sm100_fwd_out");
}
int64_t dense_decode_sm100_workspace_size(int64_t max_seq_len, int64_t num_batches, int64_t sm_count, int64_t num_kv_splits) {
  TORCH_CHECK(false, "CUDA version must be >= 12.4 for dense_decode_sm100_workspace_size");
}
#else

#define CUTLASS_CHECK(status)                                                       \
  {                                                                                 \
    cutlass::Status error = status;                                                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

using namespace cute;
using namespace cutlass::fmha::kernel;

template <bool v>
struct IsPersistent {
  static const bool value = v;
};

template <typename T, bool IsPaged128, typename PersistenceOption = IsPersistent<true>>
struct MlaSm100 {
  using Element = T;
  using ElementAcc = float;
  using ElementOut = T;

  using TileShape = Shape<_128, _128, Shape<_512, _64>>;
  using TileShapeH = cute::tuple_element_t<0, TileShape>;
  using TileShapeD = cute::tuple_element_t<2, TileShape>;

  // H K (D_latent D_rope) B
  using ProblemShape = cute::tuple<TileShapeH, int, TileShapeD, int>;

  using StrideQ = cute::tuple<int64_t, _1, int64_t>;  // H D B
  using StrideK = cute::tuple<int64_t, _1, int64_t>;  // K D B
  using StrideO = StrideK;                            // H D B
  using StrideLSE = cute::tuple<_1, int>;             // H B

  using TileScheduler =
      std::conditional_t<PersistenceOption::value, Sm100MlaPersistentTileScheduler, Sm100MlaIndividualTileScheduler>;

  using FmhaKernel = cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized<
      TileShape,
      Element,
      ElementAcc,
      ElementOut,
      ElementAcc,
      TileScheduler,
      /*kIsCpAsync=*/!IsPaged128>;
  using Fmha = cutlass::fmha::device::MLA<FmhaKernel>;
};

template <typename T>
typename T::Fmha::Arguments args_from_options(
    at::Tensor const& out,
    torch::Tensor const& lse,
    at::Tensor const& q_nope,
    at::Tensor const& q_pe,
    at::Tensor const& kv_c_and_k_pe_cache,
    at::Tensor const& seq_lens,
    at::Tensor const& page_table,
    double sm_scale,
    int64_t num_kv_splits) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q_nope.device().index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int batches = q_nope.size(0);
  int page_count_per_seq = page_table.size(1);
  int page_count_total = kv_c_and_k_pe_cache.size(0);
  int page_size = kv_c_and_k_pe_cache.size(1);
  int max_seq_len = page_size * page_count_per_seq;
  using TileShapeH = typename T::TileShapeH;
  using TileShapeD = typename T::TileShapeD;
  auto problem_shape = cute::make_tuple(TileShapeH{}, max_seq_len, TileShapeD{}, batches);

  auto [H, K, D, B] = problem_shape;
  auto [D_latent, D_rope] = D;

  float scale = float(sm_scale);

  using StrideQ = typename T::StrideQ;
  using StrideK = typename T::StrideK;
  using StrideO = typename T::StrideO;
  using StrideLSE = typename T::StrideLSE;

  StrideQ stride_Q_nope = cute::make_tuple(
      static_cast<int64_t>(q_nope.stride(1)), _1{}, static_cast<int64_t>(q_nope.stride(0)));
  StrideQ stride_Q_pe = cute::make_tuple(
      static_cast<int64_t>(q_pe.stride(1)), _1{}, static_cast<int64_t>(q_pe.stride(0)));

  StrideK stride_C = cute::make_tuple(
      static_cast<int64_t>(0 + D_latent + D_rope), _1{}, static_cast<int64_t>(page_size * (D_latent + D_rope)));
  StrideLSE stride_PT = cute::make_stride(_1{}, page_count_per_seq);
  StrideLSE stride_LSE = cute::make_tuple(_1{}, 0 + H);
  StrideO stride_O = cute::make_tuple(static_cast<int64_t>(0 + D_latent), _1{}, static_cast<int64_t>(0 + H * D_latent));

  using Element = typename T::Element;
  using ElementOut = typename T::ElementOut;
  using ElementAcc = typename T::ElementAcc;
  auto Q_nope_ptr = static_cast<Element*>(q_nope.data_ptr());
  auto Q_pe_ptr = static_cast<Element*>(q_pe.data_ptr());
  auto C_ptr = static_cast<Element*>(kv_c_and_k_pe_cache.data_ptr());
  typename T::Fmha::Arguments arguments{
      problem_shape,
      {scale,
       Q_nope_ptr,
       stride_Q_nope,
       Q_pe_ptr,
       stride_Q_pe,
       C_ptr,
       stride_C,
       C_ptr + D_latent,
       stride_C,
       static_cast<int*>(seq_lens.data_ptr()),
       static_cast<int*>(page_table.data_ptr()),
       stride_PT,
       page_count_total,
       page_size},
      {static_cast<ElementOut*>(out.data_ptr()), stride_O, static_cast<ElementAcc*>(lse.data_ptr()), stride_LSE},
      hw_info,
      // TODO(trevor-m): Change split_kv back to -1 when
      // https://github.com/NVIDIA/cutlass/issues/2274 is fixed. Split_kv=1 will
      // perform worse with larger context length and smaller batch sizes.
      static_cast<int>(num_kv_splits), // split_kv
      nullptr,       // is_var_split_kv
  };
  // TODO(kaixih@nvidia): When split_kv=-1 and is_var_split_kv=false, we compute
  // split_kv automatically based on batch size and sequence length to balance
  // workload across available SMs. Consider using var_split_kv for manual
  // control if needed.
  T::Fmha::set_split_kv(arguments);
  return arguments;
}

template <typename Element, bool IsPaged128, typename PersistenceOption>
void runMla(
    at::Tensor const& out,
    torch::Tensor const& lse,
    at::Tensor const& q_nope,
    at::Tensor const& q_pe,
    at::Tensor const& kv_c_and_k_pe_cache,
    at::Tensor const& seq_lens,
    at::Tensor const& page_table,
    at::Tensor const& workspace,
    double sm_scale,
    int64_t num_kv_splits,
    cudaStream_t stream) {
  using MlaSm100Type = MlaSm100<Element, IsPaged128, PersistenceOption>;
  typename MlaSm100Type::Fmha fmha;
  auto arguments = args_from_options<MlaSm100Type>(out, lse, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, sm_scale, num_kv_splits);

  CUTLASS_CHECK(fmha.can_implement(arguments));
  TORCH_CHECK(workspace.numel() >= static_cast<int64_t>(fmha.get_workspace_size(arguments)),
              "workspace is too small for SM100 dense MLA decode");

  CUTLASS_CHECK(fmha.initialize(arguments, workspace.data_ptr(), stream));

  CUTLASS_CHECK(fmha.run(arguments, workspace.data_ptr(), stream));
}

#define DISPATCH_BOOL(expr, const_expr, ...) \
  [&]() -> bool {                            \
    if (expr) {                              \
      constexpr bool const_expr = true;      \
      return __VA_ARGS__();                  \
    } else {                                 \
      constexpr bool const_expr = false;     \
      return __VA_ARGS__();                  \
    }                                        \
  }()

void dense_decode_sm100_fwd_out(
    torch::Tensor const& out,
    torch::Tensor const& lse,
    torch::Tensor const& q_nope,
    torch::Tensor const& q_pe,
    torch::Tensor const& kv_c_and_k_pe_cache,
    torch::Tensor const& seq_lens,
    torch::Tensor const& page_table,
    torch::Tensor const& workspace,
    double sm_scale,
    int64_t num_kv_splits) {
  TORCH_CHECK(q_nope.is_cuda(), "q_nope must be a CUDA tensor");
  const auto batch_size = q_nope.size(0);
  TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(1) == 128 && q_nope.size(2) == 512,
              "q_nope must have shape [batch, 128, 512]");
  TORCH_CHECK(q_pe.sizes() == at::IntArrayRef({batch_size, 128, 64}),
              "q_pe must have shape [batch, 128, 64]");
  TORCH_CHECK(out.sizes() == at::IntArrayRef({batch_size, 128, 512}),
              "out must have shape [batch, 128, 512]");
  TORCH_CHECK(lse.sizes() == at::IntArrayRef({batch_size, 128}) && lse.scalar_type() == at::kFloat,
              "lse must have shape [batch, 128] and dtype float32");
  TORCH_CHECK(kv_c_and_k_pe_cache.dim() == 3 && kv_c_and_k_pe_cache.size(2) == 576,
              "kv cache must have shape [pages, page_size, 576]");
  TORCH_CHECK(seq_lens.sizes() == at::IntArrayRef({batch_size}) && seq_lens.scalar_type() == at::kInt,
              "seq_lens must have shape [batch] and dtype int32");
  TORCH_CHECK(page_table.dim() == 2 && page_table.size(0) == batch_size && page_table.scalar_type() == at::kInt,
              "page_table must have shape [batch, max_pages] and dtype int32");
  TORCH_CHECK(q_nope.scalar_type() == at::kHalf || q_nope.scalar_type() == at::kBFloat16,
              "query must be float16 or bfloat16");
  TORCH_CHECK(q_pe.scalar_type() == q_nope.scalar_type() &&
              kv_c_and_k_pe_cache.scalar_type() == q_nope.scalar_type() &&
              out.scalar_type() == q_nope.scalar_type(), "query, cache, and out dtypes must match");
  TORCH_CHECK(workspace.scalar_type() == at::kByte, "workspace must have dtype uint8");
  const int device = q_nope.get_device();
  for (const at::Tensor* tensor : {&out, &lse, &q_pe, &kv_c_and_k_pe_cache, &seq_lens, &page_table, &workspace}) {
    TORCH_CHECK(tensor->is_cuda() && tensor->get_device() == device,
                "all tensors must be CUDA tensors on the same device");
  }
  TORCH_CHECK(q_nope.stride(-1) == 1 && q_pe.stride(-1) == 1, "query last dimensions must be contiguous");
  TORCH_CHECK(kv_c_and_k_pe_cache.is_contiguous() && out.is_contiguous() && lse.is_contiguous(),
              "kv cache, out, and lse must be contiguous");
  TORCH_CHECK(seq_lens.is_contiguous() && page_table.stride(-1) == 1 && workspace.is_contiguous(),
              "metadata and workspace must be contiguous");
  TORCH_CHECK(num_kv_splits == -1 || num_kv_splits > 0, "num_kv_splits must be -1 or positive");

  const auto* props = at::cuda::getDeviceProperties(q_nope.get_device());
  // The CUTLASS kernel is not validated on SM103.
  TORCH_CHECK(
      props->major == 10 && props->minor == 0,
      "dense_decode_sm100_fwd_out is only supported on compute capability 10.0, but found ",
      props->major, ".", props->minor);

  auto in_dtype = q_nope.dtype();
  at::cuda::CUDAGuard device_guard{(char)q_nope.get_device()};
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q_nope.get_device());
  const int page_size = kv_c_and_k_pe_cache.size(1);
  TORCH_CHECK(page_size > 0 && page_size <= 128 && (page_size & (page_size - 1)) == 0,
              "page_size must be a power of two no larger than 128");
  TORCH_CHECK(page_table.size(1) > 0 && page_table.size(1) % (128 / page_size) == 0,
              "page_table width must be padded to a 128-token tile");

  // NOTE(alcanderian): IsPersistent has bug with manual split_kv.
  // Kernel will hang if batch is too large with large num_kv_splits. (for example bs=8, num_kv_splits=8)
  // Maybe per batch split kv will fix this.
  DISPATCH_BOOL(page_size == 128, IsPaged128, [&] {
    DISPATCH_BOOL(num_kv_splits <= 1, NotManualSplitKV, [&] {
      if (in_dtype == at::ScalarType::Half) {
        runMla<cutlass::half_t, IsPaged128, IsPersistent<NotManualSplitKV>>(
          out, lse, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale, num_kv_splits, stream);
      } else if (in_dtype == at::ScalarType::BFloat16) {
        runMla<cutlass::bfloat16_t, IsPaged128, IsPersistent<NotManualSplitKV>>(
          out, lse, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale, num_kv_splits, stream);
      } else {
        TORCH_CHECK(false, "Unsupported input data type of MLA");
      }
      return true;
    });
    return true;
  });
}

int64_t dense_decode_sm100_workspace_size(int64_t max_seq_len, int64_t num_batches, int64_t sm_count, int64_t num_kv_splits) {
  // Workspace size depends on ElementAcc and ElementLSE (same as ElementAcc)
  // which are float, so Element type here doesn't matter.
  using MlaSm100Type = MlaSm100<cutlass::half_t, true>;

  // Get split kv. Requires problem shape and sm_count only.
  typename MlaSm100Type::Fmha::Arguments arguments;
  using TileShapeH = typename MlaSm100Type::TileShapeH;
  using TileShapeD = typename MlaSm100Type::TileShapeD;
  arguments.problem_shape =
      cute::make_tuple(TileShapeH{}, static_cast<int>(max_seq_len), TileShapeD{}, static_cast<int>(num_batches));
  // Assumes device 0 when getting sm_count.
  arguments.hw_info.sm_count =
      sm_count <= 0 ? cutlass::KernelHardwareInfo::query_device_multiprocessor_count(/*device_id=*/0) : sm_count;
  arguments.split_kv = static_cast<int>(num_kv_splits);
  MlaSm100Type::Fmha::set_split_kv(arguments);

  return MlaSm100Type::Fmha::get_workspace_size(arguments);
}

#endif
// clang-format on

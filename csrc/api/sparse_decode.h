#pragma once

#include <cstdint>
#include <limits>
#include <memory>

#include "common.h"

#include "params.h"

#include "sm90/decode/sparse_fp8/splitkv_mla.h"
#include "sm100/decode/head64/kernel.h"
#include "sm100/prefill/sparse/fwd_for_small_topk/head128/phase1.h"
#include "smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.h"
#include "smxx/decode/combine/combine.h"

// Feature set of sparse decoding kernels
enum class DecodeFeatures : int {
    HEAD_64,
    HEAD_128,

    HEAD_DIM_576,
    HEAD_DIM_512,

    V32_KVCACHE_FORMAT,
    MODEL1_KVCACHE_FORMAT,

    ATTN_SINK,
    TOPK_LENGTH,
    EXTRA_KVCACHE,
    EXTRA_TOPK_LENGTH
};

struct DecodeImplMeta {
    int num_sm_parts;
    int fixed_overhead_num_blocks;
    int block_size_topk;
};

class DecodeImplBase : public ImplBase<
    SparseAttnDecodeParams,
    DecodeFeatures
> {
public:
    virtual DecodeImplMeta get_meta(int h_q, int s_q) = 0;
};

class Decode_Sm90_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q / (h_q/64), 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            DISPATCH_NUM_HEADS(params.h_q, NUM_HEADS, [&]() {
                sm90::decode::sparse_fp8::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE, NUM_HEADS>(params);
            });
        });
    }
};

class Decode_Sm100_Head64_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_64,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q, 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE>(params);
        });
    }
};


// An implementation that calls the head64 kernel twice to process head128
// Necessary for running V3.2 shape (i.e. h = 128, d_qk = 576) on SM100f
class Decode_Sm100_Head64x2_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::HEAD_DIM_576,
        DecodeFeatures::V32_KVCACHE_FORMAT,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q, 1),
            5,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        DISPATCH_MODEL_TYPE(params.model_type, MODEL_TYPE, [&]() {
            for (int start_head_idx = 0; start_head_idx < 128; start_head_idx += 64) {
                SparseAttnDecodeParams cur_params = params;
                cur_params.q += start_head_idx * params.stride_q_h_q;
                if (cur_params.attn_sink) {
                    cur_params.attn_sink += start_head_idx;
                }
                cur_params.lse += start_head_idx;
                cur_params.out += start_head_idx * params.stride_o_h_q;
                cur_params.lse_accum += start_head_idx;
                cur_params.o_accum += start_head_idx * params.stride_o_accum_h_q;
                cur_params.h_q = 64;
                sm100::decode::head64::run_flash_splitkv_mla_fp8_sparse_kernel<MODEL_TYPE>(cur_params);
            }
        });
    }
};


class Decode_Sm100_Head128_Impl : public DecodeImplBase {
    DECLARE_SUPPORTED_FEATURES(
        DecodeFeatures::HEAD_128,
        DecodeFeatures::HEAD_DIM_512,
        DecodeFeatures::MODEL1_KVCACHE_FORMAT,
        DecodeFeatures::ATTN_SINK,
        DecodeFeatures::TOPK_LENGTH,
        DecodeFeatures::EXTRA_KVCACHE,
        DecodeFeatures::EXTRA_TOPK_LENGTH
    )

public:
    DecodeImplMeta get_meta(int h_q, int s_q) override {
        Arch arch = Arch();
        return {
            std::max(arch.num_sms / s_q / 2, 1),
            3,
            64
        };
    }

protected:
    void run_(const SparseAttnDecodeParams &params, const std::vector<FeatureT> &required_features) override {
        sm100::fwd_for_small_topk::head128::run_fwd_for_small_topk_phase1_kernel<SparseAttnFwdMode::DecodeWithSplitKV, 512>(params);
    }
};

static std::unique_ptr<DecodeImplBase> make_sparse_decode_impl(
    const Arch &arch,
    int h_q,
    int d_qk
) {
    if (arch.is_sm100f()) {
        if (h_q == 64) {
            return std::make_unique<Decode_Sm100_Head64_Impl>();
        }
        if (h_q == 128) {
            if (d_qk == 576) {
                return std::make_unique<Decode_Sm100_Head64x2_Impl>();
            }
            if (d_qk == 512) {
                return std::make_unique<Decode_Sm100_Head128_Impl>();
            }
        }
    } else if (arch.is_sm90a()) {
        return std::make_unique<Decode_Sm90_Impl>();
    }
    TORCH_CHECK(false, "Unsupported sparse decode geometry or architecture: h_q=", h_q,
                ", d_qk=", d_qk);
}

static int64_t sparse_attn_decode_workspace_size(
    int64_t b,
    int64_t s_q,
    int64_t h_q,
    int64_t d_qk,
    int64_t d_v
) {
    TORCH_CHECK(b > 0 && s_q > 0, "b and s_q must be positive");
    TORCH_CHECK(h_q == 64 || h_q == 128, "h_q must be 64 or 128");
    TORCH_CHECK(d_qk == 512 || d_qk == 576, "d_qk must be 512 or 576");
    TORCH_CHECK(d_v == 512, "d_v must be 512");
    TORCH_CHECK(b <= std::numeric_limits<int>::max() &&
                s_q <= std::numeric_limits<int>::max() &&
                h_q <= std::numeric_limits<int>::max() &&
                d_qk <= std::numeric_limits<int>::max(),
                "sparse decode dimensions exceed int32 kernel limits");

    Arch arch = Arch();
    auto impl = make_sparse_decode_impl(arch, static_cast<int>(h_q), static_cast<int>(d_qk));
    const DecodeImplMeta impl_meta = impl->get_meta(static_cast<int>(h_q), static_cast<int>(s_q));
    const int64_t total_num_splits = b + impl_meta.num_sm_parts;
    TORCH_CHECK(s_q <= std::numeric_limits<int64_t>::max() / h_q,
                "sparse decode workspace size overflows int64");
    const int64_t rows_per_split = s_q * h_q;
    TORCH_CHECK(rows_per_split <= std::numeric_limits<int64_t>::max() / (d_v + 1),
                "sparse decode workspace size overflows int64");
    const int64_t floats_per_split = rows_per_split * (d_v + 1);
    TORCH_CHECK(floats_per_split <= std::numeric_limits<int64_t>::max() / total_num_splits,
                "sparse decode workspace size overflows int64");
    const int64_t num_floats = total_num_splits * floats_per_split;
    TORCH_CHECK(num_floats <= std::numeric_limits<int64_t>::max() / static_cast<int64_t>(sizeof(float)),
                "sparse decode workspace byte size overflows int64");
    return num_floats * static_cast<int64_t>(sizeof(float));
}

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_impl(
    const at::Tensor &q,   // [b, s_q, h_q, d_qk]
    const at::Tensor &kv,   // [num_blocks, page_block_size, h_k, d_qk]
    const at::Tensor &indices,    // [b, s_q, topk]
    const std::optional<at::Tensor> &topk_length,   // [b, s_q]
    const std::optional<at::Tensor> &attn_sink, // [h_q]
    std::optional<at::Tensor> &tile_scheduler_metadata,   // num_sm_parts x (DecodingSchedMetaSize/4)
    std::optional<at::Tensor> &num_splits,                // batch_size + 1
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int d_v,
    float sm_scale,
    const std::optional<at::Tensor> &out_opt,
    const std::optional<at::Tensor> &lse_opt,
    const std::optional<at::Tensor> &workspace_opt
) {
    using bf16 = cutlass::bfloat16_t;

    // Check the architecture
    Arch arch = Arch();

    KU_CHECK_NDIM(q, 4);
    KU_CHECK_NDIM(kv, 4);
    KU_CHECK_NDIM(indices, 3);

    const int64_t b64 = q.size(0);
    const int64_t s_q64 = q.size(1);
    const int64_t h_q64 = q.size(2);
    const int64_t d_qk64 = q.size(3);
    const int64_t num_blocks64 = kv.size(0);
    const int64_t page_block_size64 = kv.size(1);
    const int64_t h_kv64 = kv.size(2);
    const int64_t topk64 = indices.size(2);
    const auto int_max = static_cast<int64_t>(std::numeric_limits<int>::max());
    TORCH_CHECK(b64 > 0 && s_q64 > 0 && h_q64 > 0 && d_qk64 > 0 &&
                num_blocks64 > 0 && page_block_size64 > 0 && h_kv64 > 0 && topk64 > 0,
                "sparse decode dimensions must be positive");
    TORCH_CHECK(b64 <= int_max && s_q64 <= int_max && h_q64 <= int_max && d_qk64 <= int_max &&
                num_blocks64 <= int_max && page_block_size64 <= int_max && h_kv64 <= int_max &&
                topk64 <= int_max,
                "sparse decode dimensions exceed int32 kernel limits");
    const int b = static_cast<int>(b64);
    const int s_q = static_cast<int>(s_q64);
    const int h_q = static_cast<int>(h_q64);
    const int d_qk = static_cast<int>(d_qk64);
    const int num_blocks = static_cast<int>(num_blocks64);
    const int page_block_size = static_cast<int>(page_block_size64);
    const int h_kv = static_cast<int>(h_kv64);
    const int topk = static_cast<int>(topk64);

    bool have_topk_length = topk_length.has_value();
    bool have_extra_kcache = extra_kv.has_value();
    bool have_extra_topk_length = extra_topk_length.has_value();
    bool have_attn_sink = attn_sink.has_value();

    int extra_num_blocks = 0, extra_page_block_size = 0, extra_topk = 0;
    if (have_extra_kcache) {
        TORCH_CHECK(extra_kv->size(0) > 0 && extra_kv->size(0) <= int_max &&
                    extra_kv->size(1) > 0 && extra_kv->size(1) <= int_max,
                    "extra sparse KV dimensions must be positive and fit int32");
        extra_num_blocks = static_cast<int>(extra_kv->size(0));
        extra_page_block_size = static_cast<int>(extra_kv->size(1));
    }
    if (extra_indices.has_value()) {
        TORCH_CHECK(extra_indices->size(-1) > 0 && extra_indices->size(-1) <= int_max,
                    "extra sparse top-k must be positive and fit int32");
        extra_topk = static_cast<int>(extra_indices->size(-1));
    }

    // metadata sanity check
    TORCH_CHECK(b > 0);
    TORCH_CHECK(s_q > 0);
    TORCH_CHECK(h_q > 0);
    TORCH_CHECK(h_kv == 1, "Currently only MQA (i.e. h_kv == 1) is supported for sparse decoding");
    TORCH_CHECK(d_qk == 576 || d_qk == 512, "Only head_size_k == 576 or 512 is supported for sparse decoding");
    TORCH_CHECK(d_v == 512, "Only head_size_v == 512 is supported for sparse decoding");
    TORCH_CHECK(topk > 0);

    if (have_extra_kcache) {
        TORCH_CHECK(extra_indices.has_value(), "extra_indices_in_kvcache must be provided when extra_kcache is provided for sparse attention");
    } else {
        TORCH_CHECK(!extra_indices.has_value(), "extra_indices_in_kvcache must not be provided when extra_k_cache is not provided");
        TORCH_CHECK(!extra_topk_length.has_value(), "extra_topk_length must not be provided when extra_k_cache is not provided");
    }

    // Check device
    KU_CHECK_DEVICE(q);
    KU_CHECK_DEVICE(kv);
    KU_CHECK_DEVICE(indices);
    KU_CHECK_DEVICE(topk_length);
    KU_CHECK_DEVICE(attn_sink);
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DEVICE(extra_kv);
    KU_CHECK_DEVICE(extra_indices);
    KU_CHECK_DEVICE(extra_topk_length);
    TORCH_CHECK(tile_scheduler_metadata.has_value() == num_splits.has_value(),
                "tile_scheduler_metadata and num_splits must either both be provided or both be absent");
    const auto q_device = q.get_device();
    TORCH_CHECK(kv.get_device() == q_device && indices.get_device() == q_device,
                "q, kv, and indices must be on the same CUDA device");
    auto check_optional_device = [q_device](const std::optional<at::Tensor> &tensor,
                                             const char *name) {
        TORCH_CHECK(!tensor.has_value() || tensor->get_device() == q_device,
                    name, " must be on the same CUDA device as q");
    };
    check_optional_device(topk_length, "topk_length");
    check_optional_device(attn_sink, "attn_sink");
    check_optional_device(tile_scheduler_metadata, "tile_scheduler_metadata");
    check_optional_device(num_splits, "num_splits");
    check_optional_device(extra_kv, "extra_kv");
    check_optional_device(extra_indices, "extra_indices");
    check_optional_device(extra_topk_length, "extra_topk_length");

    // Check data type
    KU_CHECK_DTYPE(q, torch::kBFloat16);
    TORCH_CHECK(kv.dtype() == torch::kFloat8_e4m3fn || kv.dtype() == torch::kInt8 || kv.dtype() == torch::kUInt8, "key must have dtype fp8_e4m3fn, int8 or uint8");
    if (extra_kv.has_value()) {
        TORCH_CHECK(extra_kv->dtype() == torch::kFloat8_e4m3fn || extra_kv->dtype() == torch::kInt8 || extra_kv->dtype() == torch::kUInt8, "extra k cache must have dtype fp8_e4m3fn, int8 or uint8");
    }
    KU_CHECK_DTYPE(indices, torch::kInt32);
    KU_CHECK_DTYPE(topk_length, torch::kInt32);
    KU_CHECK_DTYPE(attn_sink, torch::kFloat32);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_DTYPE(extra_indices, torch::kInt32);
    KU_CHECK_DTYPE(extra_topk_length, torch::kInt32);
    
    // Check layout
    KU_CHECK_LAST_DIM_CONTIGUOUS(q);
    KU_CHECK_LAST_DIM_CONTIGUOUS(kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(indices);
    KU_CHECK_CONTIGUOUS(topk_length);
    KU_CHECK_CONTIGUOUS(attn_sink);

    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);

    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_kv);
    KU_CHECK_LAST_DIM_CONTIGUOUS(extra_indices);
    KU_CHECK_CONTIGUOUS(extra_topk_length);
    
    // Check shape
    KU_CHECK_SHAPE(q, b, s_q, h_q, d_qk);
    {
        int bytes_per_token;
        if (d_qk == 576 && d_v == 512) {
            // V3.2 style
            bytes_per_token = 512 + 64*2 + (512/128)*4;
        } else if (d_qk == 512 && d_v == 512) {
            // MODEL1 style
            bytes_per_token = 448 + 64*2 + (448/64)*1 + 1;
        } else {
            TORCH_CHECK(false, "Unsupported head sizes for is_fp8_kvcache == True");
        }
        KU_CHECK_SHAPE(kv, num_blocks, page_block_size, h_kv, bytes_per_token);
        KU_CHECK_SHAPE(extra_kv, extra_num_blocks, extra_page_block_size, h_kv, bytes_per_token);
        TORCH_CHECK(kv.stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True for kv cache");
        if (extra_kv.has_value()) {
            TORCH_CHECK(extra_kv->stride(1) == bytes_per_token, "The whole block must be contiguous when is_fp8_cache is True for extra kv cache");
        }
    }
    KU_CHECK_SHAPE(indices, b, s_q, topk);
    KU_CHECK_SHAPE(topk_length, b);
    KU_CHECK_SHAPE(attn_sink, h_q);
    KU_CHECK_SHAPE(extra_indices, b, s_q, extra_topk);
    KU_CHECK_SHAPE(extra_topk_length, b);

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto opts = q.options();

    at::Tensor out = out_opt.has_value() ? *out_opt : torch::empty({b, s_q, h_q, d_v}, opts);
    at::Tensor lse = lse_opt.has_value() ? *lse_opt : torch::empty({b, s_q, h_q}, opts.dtype(at::kFloat));
    KU_CHECK_DEVICE(out);
    KU_CHECK_DEVICE(lse);
    KU_CHECK_DTYPE(out, torch::kBFloat16);
    KU_CHECK_DTYPE(lse, torch::kFloat32);
    KU_CHECK_CONTIGUOUS(out);
    KU_CHECK_CONTIGUOUS(lse);
    KU_CHECK_SHAPE(out, b, s_q, h_q, d_v);
    KU_CHECK_SHAPE(lse, b, s_q, h_q);
    TORCH_CHECK(out.get_device() == q_device && lse.get_device() == q_device,
                "sparse decode outputs must be on the same device as q");

    ModelType model_type;
    if (d_qk == 576) {
        model_type = ModelType::V32;
    } else if (d_qk == 512) {
        model_type = ModelType::MODEL1;
    } else {
        TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }

    std::vector<DecodeFeatures> features;
    if (h_q == 64) {
        features.push_back(DecodeFeatures::HEAD_64);
    } else if (h_q == 128) {
        features.push_back(DecodeFeatures::HEAD_128);
    } else {
        TORCH_CHECK(false, "Unsupported h_q: ", h_q);
    }
    if (d_qk == 576) {
        features.push_back(DecodeFeatures::HEAD_DIM_576);
    } else if (d_qk == 512) {
        features.push_back(DecodeFeatures::HEAD_DIM_512);
    } else {
        TORCH_CHECK(false, "Unsupported d_qk: ", d_qk);
    }
    if (model_type == ModelType::V32) {
        features.push_back(DecodeFeatures::V32_KVCACHE_FORMAT);
    } else if (model_type == ModelType::MODEL1) {
        features.push_back(DecodeFeatures::MODEL1_KVCACHE_FORMAT);
    } else {
        TORCH_CHECK(false, "Unsupported model type: ", (int)model_type);
    }
    if (have_attn_sink) {
        features.push_back(DecodeFeatures::ATTN_SINK);
    }
    if (have_topk_length) {
        features.push_back(DecodeFeatures::TOPK_LENGTH);
    }
    if (have_extra_kcache) {
        features.push_back(DecodeFeatures::EXTRA_KVCACHE);
    }
    if (have_extra_topk_length) {
        features.push_back(DecodeFeatures::EXTRA_TOPK_LENGTH);
    }

    auto impl = make_sparse_decode_impl(arch, h_q, d_qk);

    DecodeImplMeta impl_meta = impl->get_meta(h_q, s_q);

    SparseAttnDecodeParams params = {
        b, s_q, h_q, h_kv, d_qk, d_v,
        sm_scale, sm_scale * LOG_2_E,
        num_blocks, page_block_size, topk,
        model_type,

        (bf16*)q.data_ptr(),
        (bf16*)kv.data_ptr(),
        (int*)indices.data_ptr(),
        ku::get_optional_tensor_ptr<int>(topk_length),
        ku::get_optional_tensor_ptr<float>(attn_sink),
        (float*)lse.data_ptr(),
        (bf16*)out.data_ptr(),

        extra_num_blocks, extra_page_block_size, extra_topk,
        ku::get_optional_tensor_ptr<bf16>(extra_kv),
        ku::get_optional_tensor_ptr<int>(extra_indices),
        ku::get_optional_tensor_ptr<int>(extra_topk_length),

        int64_stride_to_int(q.stride(0)), int64_stride_to_int(q.stride(1)), int64_stride_to_int(q.stride(2)),
        int64_stride_to_int(kv.stride(0)), int64_stride_to_int(kv.stride(1)),
        int64_stride_to_int(indices.stride(0)), int64_stride_to_int(indices.stride(1)),
        int64_stride_to_int(lse.stride(0)), int64_stride_to_int(lse.stride(1)),
        int64_stride_to_int(out.stride(0)), int64_stride_to_int(out.stride(1)), int64_stride_to_int(out.stride(2)),

        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_kv->stride(1)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(0)) : 0,
        have_extra_kcache ? int64_stride_to_int(extra_indices->stride(1)) : 0,
        at::cuda::getCurrentCUDAStream().stream()
    };

    // Get MLA metadata if necessary
    at::Tensor o_accum, lse_accum;
    if (!tile_scheduler_metadata.has_value()) {
        tile_scheduler_metadata = torch::empty({impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/4}, opts.dtype(torch::kInt32));
        num_splits = torch::empty({b+1}, opts.dtype(torch::kInt32));
        KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
        KU_CHECK_CONTIGUOUS(num_splits);

        GetDecodeSchedMetaParams get_sched_meta_params = {
            b, s_q,
            impl_meta.block_size_topk,
            impl_meta.fixed_overhead_num_blocks,
            topk,
            extra_topk,
            ku::get_optional_tensor_ptr<int>(topk_length),
            ku::get_optional_tensor_ptr<int>(extra_topk_length),
            nullptr,
            (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr(),
            num_splits->data_ptr<int>(),
            impl_meta.num_sm_parts,
            at::cuda::getCurrentCUDAStream().stream()
        };
        smxx::decode::run_get_decoding_sched_meta_kernel(get_sched_meta_params);
    }
    // Stick the metadata pointers to `params`
    KU_CHECK_DEVICE(tile_scheduler_metadata);
    KU_CHECK_DEVICE(num_splits);
    KU_CHECK_DTYPE(tile_scheduler_metadata, torch::kInt32);
    KU_CHECK_DTYPE(num_splits, torch::kInt32);
    KU_CHECK_CONTIGUOUS(tile_scheduler_metadata);
    KU_CHECK_CONTIGUOUS(num_splits);
    KU_CHECK_SHAPE(tile_scheduler_metadata, impl_meta.num_sm_parts, sizeof(DecodingSchedMeta)/sizeof(int));
    KU_CHECK_SHAPE(num_splits, b+1);
    params.tile_scheduler_metadata_ptr = (DecodingSchedMeta*)tile_scheduler_metadata->data_ptr();
    params.num_splits_ptr = num_splits->data_ptr<int>();
    params.num_sm_parts = impl_meta.num_sm_parts;

    // Allocate intermediate buffers for split-KV, or carve them out of the
    // caller-owned byte workspace.  The latter is the CUDA-graph path: no
    // allocator call occurs between launch and combine.
    const int64_t total_num_splits = static_cast<int64_t>(b) + impl_meta.num_sm_parts;
    TORCH_CHECK(static_cast<int64_t>(s_q) <= std::numeric_limits<int64_t>::max() / h_q,
                "sparse decode accumulation workspace size overflows int64");
    const int64_t rows_per_split = static_cast<int64_t>(s_q) * h_q;
    TORCH_CHECK(total_num_splits <= std::numeric_limits<int64_t>::max() / rows_per_split,
                "sparse decode accumulation workspace size overflows int64");
    const int64_t lse_accum_numel = total_num_splits * rows_per_split;
    TORCH_CHECK(lse_accum_numel <= std::numeric_limits<int64_t>::max() / (d_v + 1),
                "sparse decode accumulation workspace size overflows int64");
    const int64_t workspace_floats = lse_accum_numel * (d_v + 1);
    if (workspace_opt.has_value()) {
        const at::Tensor &workspace = *workspace_opt;
        KU_CHECK_DEVICE(workspace);
        KU_CHECK_DTYPE(workspace, torch::kUInt8);
        KU_CHECK_CONTIGUOUS(workspace);
        TORCH_CHECK(workspace_floats <=
                        std::numeric_limits<int64_t>::max() / static_cast<int64_t>(sizeof(float)),
                    "sparse decode accumulation workspace byte size overflows int64");
        const int64_t required_bytes = workspace_floats * sizeof(float);
        TORCH_CHECK(workspace.numel() >= required_bytes,
                    "sparse decode workspace has ", workspace.numel(),
                    " bytes, but ", required_bytes, " are required");
        TORCH_CHECK(workspace.get_device() == q.get_device(),
                    "sparse decode workspace must be on the same device as q");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % alignof(float) == 0,
                    "sparse decode workspace must be aligned for float accumulation");
        params.lse_accum = reinterpret_cast<float *>(workspace.data_ptr<uint8_t>());
        params.o_accum = params.lse_accum + lse_accum_numel;
        params.stride_lse_accum_split = int64_stride_to_int(static_cast<int64_t>(s_q) * h_q);
        params.stride_lse_accum_s_q = int64_stride_to_int(h_q);
        params.stride_o_accum_split = int64_stride_to_int(static_cast<int64_t>(s_q) * h_q * d_v);
        params.stride_o_accum_s_q = int64_stride_to_int(static_cast<int64_t>(h_q) * d_v);
        params.stride_o_accum_h_q = d_v;
    } else {
        lse_accum = torch::empty({total_num_splits, s_q, h_q}, opts.dtype(at::kFloat));
        o_accum = torch::empty({total_num_splits, s_q, h_q, d_v}, opts.dtype(at::kFloat));
        KU_CHECK_CONTIGUOUS(lse_accum);
        KU_CHECK_CONTIGUOUS(o_accum);
        params.lse_accum = lse_accum.data_ptr<float>();
        params.o_accum = o_accum.data_ptr<float>();
        params.stride_lse_accum_split = int64_stride_to_int(lse_accum.stride(0));
        params.stride_lse_accum_s_q = int64_stride_to_int(lse_accum.stride(1));
        params.stride_o_accum_split = int64_stride_to_int(o_accum.stride(0));
        params.stride_o_accum_s_q = int64_stride_to_int(o_accum.stride(1));
        params.stride_o_accum_h_q = int64_stride_to_int(o_accum.stride(2));
    }

    impl->run(params, features);
    
    CombineParams combine_params = {
        b, s_q, h_q, d_v,

        params.lse,
        params.out,
        params.stride_lse_b, params.stride_lse_s_q,
        params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q,

        params.lse_accum,
        params.o_accum,
        params.stride_lse_accum_split, params.stride_lse_accum_s_q,
        params.stride_o_accum_split, params.stride_o_accum_s_q, params.stride_o_accum_h_q,

        params.tile_scheduler_metadata_ptr,
        params.num_splits_ptr,
        params.num_sm_parts,

        ku::get_optional_tensor_ptr<float>(attn_sink),
        at::cuda::getCurrentCUDAStream().stream()
    };
    smxx::decode::run_flash_mla_combine_kernel<bf16>(combine_params);

    return {out, lse.transpose(1, 2), tile_scheduler_metadata, num_splits};
}

static std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>>
sparse_attn_decode_interface(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    std::optional<at::Tensor> &tile_scheduler_metadata,
    std::optional<at::Tensor> &num_splits,
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int d_v,
    float sm_scale
) {
    return sparse_attn_decode_impl(
        q, kv, indices, topk_length, attn_sink, tile_scheduler_metadata, num_splits,
        extra_kv, extra_indices, extra_topk_length, d_v, sm_scale,
        std::nullopt, std::nullopt, std::nullopt);
}

static std::tuple<at::Tensor, at::Tensor> sparse_attn_decode_out_interface(
    const at::Tensor &out,
    const at::Tensor &lse,
    const at::Tensor &workspace,
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    const std::optional<at::Tensor> &topk_length,
    const std::optional<at::Tensor> &attn_sink,
    const std::optional<at::Tensor> &tile_scheduler_metadata,
    const std::optional<at::Tensor> &num_splits,
    const std::optional<at::Tensor> &extra_kv,
    const std::optional<at::Tensor> &extra_indices,
    const std::optional<at::Tensor> &extra_topk_length,
    int d_v,
    float sm_scale
) {
    std::optional<at::Tensor> metadata = tile_scheduler_metadata;
    std::optional<at::Tensor> splits = num_splits;
    auto result = sparse_attn_decode_impl(
        q, kv, indices, topk_length, attn_sink, metadata, splits,
        extra_kv, extra_indices, extra_topk_length, d_v, sm_scale,
        out, lse, workspace);
    TORCH_CHECK(std::get<2>(result).has_value() && std::get<3>(result).has_value(),
                "sparse decode scheduler initialization did not produce metadata");
    return {*std::get<2>(result), *std::get<3>(result)};
}

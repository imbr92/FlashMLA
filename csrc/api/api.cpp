#include <pybind11/pybind11.h>

#include "sparse_fwd.h"
#include "sparse_decode.h"
#include "dense_decode.h"
#include "dense_decode_sm100.h"
#include "dense_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMLA";
    m.def("sparse_decode_fwd", &sparse_attn_decode_interface);
    m.def("sparse_decode_fwd_out", &sparse_attn_decode_out_interface);
    m.def("sparse_decode_fwd_workspace_size", &sparse_attn_decode_workspace_size);
    m.def("dense_decode_fwd", &dense_attn_decode_interface);
    m.def("dense_decode_sm100_fwd_out", &dense_decode_sm100_fwd_out);
    m.def("dense_decode_sm100_workspace_size", &dense_decode_sm100_workspace_size);
    m.def("sparse_prefill_fwd", &sparse_attn_prefill_interface);
    m.def("sparse_prefill_fwd_out", &sparse_attn_prefill_out_interface);
    m.def("dense_prefill_fwd", &FMHACutlassSM100FwdRun);
    m.def("dense_prefill_bwd", &FMHACutlassSM100BwdRun);
}

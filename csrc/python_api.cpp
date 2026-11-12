#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "buffers/base.hpp"
#include "buffers/bucket.hpp"
#include "buffers/engram.hpp"
#include "buffers/ep.hpp"
#include "buffers/pp.hpp"
#include "comm/api.hpp"
#include "runtime/jit.hpp"
#include "utils/event.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    using namespace deep_ep;

    m.doc() = "DeepEP communication buffers (Ascend)";
    config::register_apis(m);
    m.attr("topk_idx_t") = py::cast(torch::kInt64);

    py::class_<EventHandle>(m, "EventHandle")
        .def(py::init<>())
        .def("current_stream_wait", &EventHandle::current_stream_wait);

    comm::register_apis(m);
    base::register_apis(m);

    py::class_<EPBuffer, BufferBase>(m, "EPBuffer")
        .def(py::init<int, int, std::string, int64_t, int64_t, bool, int, int, bool>())
        .def_readonly("context", &EPBuffer::context)
        .def_readonly("lb_storage", &EPBuffer::lb_storage)
        .def("lb_prefetch_weights", &EPBuffer::lb_prefetch_weights)
        .def("lb_reduce_grads", &EPBuffer::lb_reduce_grads)
        .def("dispatch", &EPBuffer::dispatch,
             py::arg("x"), py::arg("sf"), py::arg("topk_idx"), py::arg("topk_weights"),
             py::arg("cumulative_local_expert_recv_stats"), py::arg("cached_num_recv_tokens"),
             py::arg("cached_num_expanded_tokens"), py::arg("cached_num_recv_tokens_per_expert_list"),
             py::arg("cached_psum_num_recv_tokens_per_rank"), py::arg("cached_psum_num_recv_tokens_per_expert"),
             py::arg("cached_num_unaligned_recv_tokens_per_expert"), py::arg("cached_dst_buffer_slot_idx"),
             py::arg("cached_dst_gsge_idx"), py::arg("cached_recv_src_metadata"),
             py::arg("num_max_tokens_per_rank"), py::arg("num_experts"), py::arg("expert_alignment"),
             py::arg("num_ai_cores"), py::arg("previous_event"), py::arg("async_with_compute_stream"),
             py::arg("allocate_on_comm_stream"), py::arg("do_handle_copy"), py::arg("do_cpu_sync"),
             py::arg("do_expand"), py::arg("do_zero_padding"), py::arg("use_cube_aligned_col_major_sf"),
             py::arg("defer_epilogue"), py::arg("decode_handle") = false)
        .def("combine", &EPBuffer::combine,
             py::arg("x"), py::arg("topk_weights"), py::arg("bias_0"), py::arg("bias_1"),
             py::arg("src_metadata"), py::arg("combined_topk_idx"), py::arg("psum_num_recv_tokens_per_rank"),
             py::arg("num_experts"), py::arg("num_max_tokens_per_rank"), py::arg("num_sms"),
             py::arg("previous_event"), py::arg("async_with_compute_stream"), py::arg("allocate_on_comm_stream"),
             py::arg("defer_epilogue"), py::arg("decode_handle") = false);

    py::class_<PPBuffer, BufferBase>(m, "PPBuffer")
        .def(py::init<int, int, std::string, int64_t, int, bool, int, bool>())
        .def_readonly("context", &PPBuffer::context)
        .def_readonly("storage", &PPBuffer::storage)
        .def("send", &PPBuffer::send)
        .def("recv", &PPBuffer::recv);

    py::class_<EngramBuffer, BufferBase>(m, "EngramBuffer")
        .def(py::init<int, int, std::string, int64_t, int64_t, bool, int, bool>())
        .def_readonly("context", &EngramBuffer::context)
        .def("set_config", &EngramBuffer::set_config)
        .def("write", &EngramBuffer::write)
        .def("fetch", &EngramBuffer::fetch);

    py::class_<BucketBuffer, BufferBase>(m, "BucketBuffer")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<std::string>, int64_t, bool, int, bool>())
        .def_readonly("contexts", &BucketBuffer::contexts)
        .def_readonly("storage", &BucketBuffer::storage)
        .def("reduce_scatter", &BucketBuffer::reduce_scatter)
        .def("all_reduce", &BucketBuffer::all_reduce)
        .def("all_gather", &BucketBuffer::all_gather);

    m.def("calculate_ep_buffer_size", &EPBuffer::calculate_buffer_size);
    m.def("get_num_allocation_alignment", []() {
        return kNumAllocationAlignmentBytes;
    });
    m.def("get_num_tma_alignment", []() {
        return kNumTMAAlignmentBytes;
    });
    m.def("get_num_rdma_alignment", []() {
        return kNumRDMAAlignmentBytes;
    });
}

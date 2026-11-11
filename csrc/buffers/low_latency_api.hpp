#pragma once

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_layout.h>
#include <deep_ep/impls/low_latency/dispatch/low_latency/common/ep_memory_server_dedup_tiling.h>
#include <deep_ep/impls/low_latency/combine/low_latency/combine_layout.h>
#include <deep_ep/impls/low_latency/combine/low_latency/ep_memory_server_dedup_combine_tiling.h>

#include "low_latency.hpp"
#include "../kernels/ep/low_latency.hpp"

namespace deep_ep::low_latency {

namespace py = pybind11;
using Shape = std::vector<int64_t>;

static uint8_t* tensor_ptr(const py::dict& tensors, const char* name, const Shape& shape, const char* dtype) {
    const auto value = tensors[name];
    const auto torch = py::module_::import("torch");
    if (not torch.attr("is_tensor")(value).cast<bool>())
        throw std::invalid_argument("LL requires tensors");
    if (not value.attr("is_contiguous")().cast<bool>() or value.attr("shape").cast<Shape>() != shape or
        not value.attr("dtype").is(torch.attr(dtype)))
        throw std::invalid_argument("LL tensor shape, dtype or contiguity mismatch");
    const auto device = value.attr("device");
    const auto type = device.attr("type").cast<std::string>();
    if ((type != "npu" and type != "privateuseone") or
        device.attr("index").cast<int>() != c10_npu::getCurrentNPUStream().device_index())
        throw std::invalid_argument("LL tensors must be on the current NPU");
    return reinterpret_cast<uint8_t*>(value.attr("data_ptr")().cast<uintptr_t>());
}

static DispatchDedup::EpMemoryServerDedupLayout get_dispatch_layout(int num_ranks, int num_tokens, int hidden, int num_topk,
                                                                   int num_experts, bool fp8, bool native_dispatch) {
    DispatchDedup::EpMemoryServerDedupLayout result{};
    using InputMode = DispatchDedup::EpServerDedupInputMode;
    const auto mode = fp8 ? InputMode::PrequantizedFp8Packs : InputMode::Plain16;
    if (num_tokens <= 0 or (not native_dispatch and num_tokens > 256) or
        DispatchDedup::BuildDispatchServerDedupLayout(num_ranks, num_tokens, hidden, num_topk, num_experts, &result, mode) !=
            DispatchDedup::DISPATCH_DEDUP_SUCCESS or
        not DispatchDedup::DispatchDedupPacketUbFits(result, num_tokens, num_topk, num_experts, num_ranks, native_dispatch) or
        (not native_dispatch and result.capacityRows > CombineDedup::kEpServerDedupCombineMaxTokenNum) or
        (native_dispatch and static_cast<int64_t>(num_ranks) * num_tokens > std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("LL dispatch exceeds topology, shape, UB or workspace limits");
    return result;
}

template <typename T>
static py::bytes tiling_bytes(const T& data) {
    return py::bytes(reinterpret_cast<const char*>(&data), sizeof(data));
}

static uint8_t dtype_code(const std::string& dtype) {
    if (dtype == "bfloat16")
        return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_BFP16;
    if (dtype == "float16")
        return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP16;
    if (dtype == "float8_e4m3fn")
        return DispatchDedup::DISPATCH_DEDUP_DATA_TYPE_FP8E4M3;
    throw std::invalid_argument("LL dispatch supports BF16/FP16 or packed FP8 E4M3FN");
}

static void validate_capacity(int num_tokens, int capacity, bool native_dispatch = false) {
    if (num_tokens <= 0 or num_tokens > capacity or (not native_dispatch and capacity > 256))
        throw std::invalid_argument("LL input exceeds the dispatch capacity");
    if (native_dispatch and num_tokens != capacity)
        throw std::invalid_argument("Native LL dispatch requires input padded to the common capacity");
}

static py::object get_layout(int num_ranks, int num_tokens, int hidden, int num_topk, int num_experts,
                             const std::string& dtype, bool native_dispatch) {
    // Unsupported shapes return None so the caller can select a different transport.
    dtype_code(dtype);
    try {
        const auto plan = get_dispatch_layout(num_ranks, num_tokens, hidden, num_topk, num_experts,
                                              dtype == "float8_e4m3fn", native_dispatch);
        py::dict result;
        result["capacity_rows"] = plan.capacityRows;
        return result;
    } catch (const std::invalid_argument&) {
        return py::none();
    }
}

static py::bytes get_dispatch_tiling(int num_ranks, int num_tokens, int hidden, int num_topk, int num_experts,
                                     const std::string& dtype, int num_max_tokens_per_rank, bool native_dispatch) {
    validate_capacity(num_tokens, num_max_tokens_per_rank, native_dispatch);
    const bool fp8 = dtype == "float8_e4m3fn";
    const auto plan = get_dispatch_layout(num_ranks, num_tokens, hidden, num_topk, num_experts, fp8, native_dispatch);
    DispatchDedup::EpServerDedupTilingData data{};
    data.bs = num_tokens;
    data.h = hidden;
    data.numMaxTokensPerRank = num_max_tokens_per_rank;
    data.topK = num_topk;
    data.moeExpertNum = num_experts;
    data.expertTokenNumsType = 1;
    data.dtype = dtype_code(dtype);
    data.quantMode = fp8 ? 4 : 0;
    data.inputMode = static_cast<uint8_t>(plan.inputMode);
    data.expandXOutDtype = data.dtype;
    return tiling_bytes(data);
}

static py::bytes get_combine_tiling(int num_ranks, int num_tokens, int hidden, int num_topk, int num_experts,
                                    const std::string& dtype, int num_max_tokens_per_rank) {
    validate_capacity(num_tokens, num_max_tokens_per_rank);
    CombineDedup::Layout plan{};
    const auto ub_bytes = get_low_latency_combine_ub_bytes();
    if ((dtype != "bfloat16" and dtype != "float16") or
        not CombineDedup::BuildLayout(num_ranks, num_tokens, hidden, num_topk, num_experts, plan, ub_bytes))
        throw std::invalid_argument("LL combine exceeds dtype, topology, UB or workspace limits");
    CombineDedup::EpServerDedupCombineTilingData data{};
    data.bs = num_tokens;
    data.h = hidden;
    data.numMaxTokensPerRank = num_max_tokens_per_rank;
    data.topK = num_topk;
    data.moeExpertNum = num_experts;
    data.dtype = dtype_code(dtype);
    data.recvUbBytes = ub_bytes;
    return tiling_bytes(data);
}

static void dispatch(LowLatencyBuffer& session, const py::dict& tensors, int num_tokens, int hidden,
                     int num_topk, int num_experts, int rows, const std::string& dtype) {
    dtype_code(dtype);
    const bool fp8 = dtype == "float8_e4m3fn";
    const bool native_dispatch = tensors.contains("native_source_ids") and not tensors["native_source_ids"].is_none();
    const auto plan = get_dispatch_layout(session.world(), num_tokens, hidden, num_topk, num_experts, fp8, native_dispatch);
    const int64_t capacity_rows = plan.capacityRows;
    if (rows < capacity_rows or session.capacity() != DispatchDedup::kEpServerDedupPeerBytes)
        throw std::invalid_argument("Invalid LL output capacity, workspace or stream");

    deepep::EpServerDedupKernelArgs args{};
    args.commArgs = static_cast<uint8_t*>(session.comm_args.data_ptr());
    args.x = tensor_ptr(tensors, "input", {num_tokens, hidden}, dtype.c_str());
    args.expertIds = tensor_ptr(tensors, "indices", {num_tokens, num_topk}, "int32");
    args.expertScales = tensor_ptr(tensors, "weights", {num_tokens, num_topk}, "float32");
    args.expandXOut = tensor_ptr(tensors, "output", {rows, hidden}, dtype.c_str());
    // Each group of 128 elements carries one FP32 scale or four packed E8M0 scales.
    if (fp8) {
        const auto num_scale_packs = (hidden + 127) / 128;
        const auto torch = py::module_::import("torch");
        const auto scale_dtype = tensors["scales"].attr("dtype").is(torch.attr("float32")) ? "float32" : "int32";
        args.dynamicScalesOut = tensor_ptr(tensors, "output_scales", {rows, num_scale_packs}, scale_dtype);
        args.inputScales = tensor_ptr(tensors, "scales", {num_tokens, num_scale_packs}, scale_dtype);
    }
    args.expertTokenNumsOut = tensor_ptr(tensors, "expert_recv_counts", {num_experts / session.world()}, "int64");
    args.sendCountsOut = tensor_ptr(tensors, "send_counts", {num_experts}, "int32");
    args.tokenTypeOut = tensor_ptr(tensors, "token_type", {capacity_rows}, "int32");
    args.destinationIndexOut = tensor_ptr(tensors, "destination_index", {capacity_rows, 3}, "int32");
    args.relayReadIndexOut = tensor_ptr(tensors, "relay_read_index", {capacity_rows, num_topk, 2}, "int32");
    args.sourceMaskOut = tensor_ptr(tensors, "source_mask", {num_tokens}, "int32");
    args.expertScalesOut = tensor_ptr(tensors, "source_weights", {num_tokens, num_topk}, "float32");
    args.tilingData = tensor_ptr(tensors, "tiling", {sizeof(DispatchDedup::EpServerDedupTilingData)}, "uint8");
    args.nativeSourceIds = native_dispatch ? tensor_ptr(tensors, "native_source_ids", {capacity_rows}, "int32") : nullptr;
    session.check_available();
    py::gil_scoped_release release;
    launch_dispatch_low_latency(args);
}

static void combine(LowLatencyBuffer& session, const py::dict& tensors, int num_tokens, int hidden,
                    int num_topk, int num_experts, const std::string& dtype) {
    CombineDedup::Layout plan{};
    if (session.capacity() != DispatchDedup::kEpServerDedupPeerBytes or (dtype != "bfloat16" and dtype != "float16") or
        not CombineDedup::BuildLayout(session.world(), num_tokens, hidden, num_topk, num_experts, plan, session.combine_ub_bytes))
        throw std::invalid_argument("Invalid LL combine dtype, shape, workspace or stream");
    const int64_t capacity_rows = plan.capacityRows;
    deepep::EpServerDedupCombineKernelArgs args{};
    args.commArgs = static_cast<uint8_t*>(session.comm_args.data_ptr());
    // Python validates rows against the dispatch handle. The device selects
    // rows below the actual receive count. Metadata uses the full capacity.
    const auto input_shape = tensors["input"].attr("shape").cast<Shape>();
    if (input_shape.size() != 2 or input_shape[0] < 0 or input_shape[1] != hidden)
        throw std::invalid_argument("LL combine input must have shape [rows, H]");
    args.expertOut = tensor_ptr(tensors, "input", input_shape, dtype.c_str());
    args.destinationIndex = tensor_ptr(tensors, "destination_index", {capacity_rows, 3}, "int32");
    args.sendCounts = tensor_ptr(tensors, "send_counts", {num_experts}, "int32");
    args.tokenType = tensor_ptr(tensors, "token_type", {capacity_rows}, "int32");
    args.relayReadIndex = tensor_ptr(tensors, "relay_read_index", {capacity_rows, num_topk, 2}, "int32");
    args.sourceMask = tensor_ptr(tensors, "source_mask", {num_tokens}, "int32");
    args.yOut = tensor_ptr(tensors, "output", {num_tokens, hidden}, dtype.c_str());
    args.tilingData = tensor_ptr(tensors, "tiling", {sizeof(CombineDedup::EpServerDedupCombineTilingData)}, "uint8");
    session.check_available();
    py::gil_scoped_release release;
    launch_combine_low_latency(args);
}

static void register_apis(py::module_& m) {
    m.def("low_latency_layout", &get_layout,
          py::arg("p"), py::arg("b"), py::arg("h"), py::arg("k"), py::arg("e"), py::arg("dtype"),
          py::arg("native_dispatch") = false);
    m.def("low_latency_dispatch_tiling", &get_dispatch_tiling,
          py::arg("p"), py::arg("b"), py::arg("h"), py::arg("k"), py::arg("e"), py::arg("dtype"),
          py::arg("num_max_tokens_per_rank"), py::arg("native_dispatch") = false);
    m.def("low_latency_combine_tiling", &get_combine_tiling);
    m.def("dispatch_low_latency", &dispatch);
    m.def("combine_low_latency", &combine);
}

}  // namespace deep_ep::low_latency

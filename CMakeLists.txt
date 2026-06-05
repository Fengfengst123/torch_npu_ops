include(cc_library)

set(TORCH_NPU_KERNELS_HEADERS
  custom_functions_npu/atb_common.h
  custom_functions_npu/operation_cache_compute.h
  custom_functions_npu/operation_create.h
  custom_functions_npu/utils.h
  ops_npu/npu_ops.h
)

set(TORCH_NPU_KERNELS_SOURCES
  custom_functions_npu/atb_common.cpp
  custom_functions_npu/operation_cache_compute.cpp
  custom_functions_npu/utils.cpp
  ops_npu/paged_attention_atb.cpp
  ops_npu/reshape_and_cach_atb.cpp
  ops_npu/self_attention_atb.cpp
  ops_npu/rope_atb.cpp
)


set(TRITON_NPU_API_SRCS
  triton_npu/torch_api/npu_triton_l2norm.cpp
  triton_npu/torch_api/npu_triton_rope_inplace.cpp
  triton_npu/torch_api/npu_triton_fused_gdn_gating.cpp
  triton_npu/torch_api/npu_triton_layernorm_fwd.cpp
  triton_npu/torch_api/npu_triton_causal_conv1d_update.cpp
  triton_npu/torch_api/npu_triton_causal_conv1d_update_v2.cpp
  triton_npu/torch_api/npu_triton_recurrent_gated_delta_rule.cpp
  triton_npu/torch_api/npu_triton_fused_sigmoid_gating_delta_rule_update.cpp
  triton_npu/torch_api/npu_triton_chunk_gated_delta_rule.cpp
  triton_npu/torch_api/npu_triton_fused_qkvzba_split_reshape.cpp
  triton_npu/torch_api/npu_triton_split_rmsnorm_rope.cpp
)

cc_library(
  NAME
    triton_adapter
  HDRS
    triton_npu/kernel_registry.h
    triton_npu/operation_base.h
    triton_npu/args_builder.h
    triton_npu/operations.h
    triton_npu/torch_api/operation_factory.h
    triton_npu/torch_api/triton_ops_api.h
  SRCS
    triton_npu/kernel_registry.cpp
    ${TRITON_NPU_API_SRCS}
  DEPS
    "$ENV{ASCEND_HOME_PATH}/lib64/libruntime.so"
    ascendcl
    nlohmann_json::nlohmann_json
    glog::glog
)

if(DEFINED ENV{TRITON_BINARY_PATH} AND NOT "$ENV{TRITON_BINARY_PATH}" STREQUAL "")
  set(TRITON_BIN_PATH "$ENV{TRITON_BINARY_PATH}")
else()
  set(TRITON_BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/triton_npu/binary")
endif()

message(STATUS "TRITON_BINARY_PATH = ${TRITON_BIN_PATH}")

target_compile_definitions(triton_adapter PUBLIC
  TRITON_BINARY_PATH="${TRITON_BIN_PATH}"
)

target_include_directories(triton_adapter PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_npu
)

cc_library(
  NAME
    torch_npu_kernels
  HDRS
    ${TORCH_NPU_KERNELS_HEADERS}
  SRCS
    ${TORCH_NPU_KERNELS_SOURCES}
  DEPS
    glog::glog
    torch
    torch_npu
    atb_customize
)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

file(GLOB_RECURSE TRITON_KERNEL_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/triton_npu/triton_src/*.py"
)

set(TRITON_NPU_STAMP_FILE "${CMAKE_CURRENT_BINARY_DIR}/triton_npu_binaries.stamp")

add_custom_command(
  OUTPUT ${TRITON_NPU_STAMP_FILE}
  COMMAND ${CMAKE_COMMAND} -E env
    TRITON_BINARY_PATH=${TRITON_BIN_PATH}
    ${Python3_EXECUTABLE} setup.py
  COMMAND ${CMAKE_COMMAND} -E touch ${TRITON_NPU_STAMP_FILE}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/triton_npu
  COMMENT "Generating Triton NPU kernel binaries into ${TRITON_BIN_PATH}"
  DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_npu/setup.py
    ${TRITON_KERNEL_SOURCES}
)

add_custom_target(triton_npu_binaries
  DEPENDS ${TRITON_NPU_STAMP_FILE}
)

add_dependencies(triton_adapter triton_npu_binaries)

target_include_directories(torch_npu_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(torch_npu_kernels PRIVATE
  TORCH_NPU_KERNELS_VERSION="1.0.0"
)

# Add ascendc_npu subdirectory (if it exists)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/ascendc_npu)
  add_subdirectory(ascendc_npu)
endif()


add_subdirectory(triton_npu/test)

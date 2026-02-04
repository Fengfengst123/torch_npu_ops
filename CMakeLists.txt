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

target_include_directories(torch_npu_kernels PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(torch_npu_kernels PRIVATE
  TORCH_NPU_KERNELS_VERSION="1.0.0"
)

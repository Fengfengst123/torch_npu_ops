include(cc_library)

cc_library(
  NAME
    ascendc_ops
  HDRS
    ascendc_ops_api.h
    pytorch_npu_helper.hpp
  SRCS
    gemma_rms_norm.cpp
    layer_norm_fwd.cpp
  DEPS
    torch
    torch_npu
    ascendcl
    glog::glog
)

target_include_directories(ascendc_ops PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

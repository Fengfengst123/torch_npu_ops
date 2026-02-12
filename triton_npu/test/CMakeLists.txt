include(cc_test)

get_filename_component(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(BINARY_DIR "${TEST_DIR}/binary")

cc_test(
  NAME
    rope_inplace_test
  SRCS
    triton_rope_inplace_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    triton_adapter
    torch
    torch_npu
)

target_compile_definitions(rope_inplace_test PRIVATE
  TEST_BINARY_DIR="${BINARY_DIR}"
)

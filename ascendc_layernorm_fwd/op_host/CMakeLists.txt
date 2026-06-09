# Copyright 2026 The xLLM Authors. All Rights Reserved.

add_modules_sources(
  HOSTNAME ${OPHOST_NAME}
  MODE PRIVATE
  DIR ${CMAKE_CURRENT_SOURCE_DIR}
  OPTYPE layer_norm_fwd
  ACLNNTYPE aclnn
)

import os
import glob
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension

import torch_npu
from torch_npu.utils.cpp_extension import NpuExtension

PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
USE_NINJA = os.getenv('USE_NINJA') == '1'
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

source_files = [
    os.path.join(BASE_DIR, "op_register.cpp"),
    os.path.join(BASE_DIR, "..", "ascendc_npu", "gemma_rms_norm.cpp"),
]

exts = []
ext = NpuExtension(
    name="npu_python_extension_lib",
    sources=source_files,
    extra_compile_args = [
        '-I' + os.path.join(PYTORCH_NPU_INSTALL_PATH, "include/third_party/acl/inc"),
        '-I' + os.path.join(BASE_DIR, "..", "ascendc_npu"),  # 添加 pytorch_npu_helper.hpp 和 ascendc_ops_api.h 路径
    ],
)
exts.append(ext)

setup(
    name="npu_python_extension",
    version='1.0',
    keywords='npu_python_extension',
    ext_modules=exts,
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
)

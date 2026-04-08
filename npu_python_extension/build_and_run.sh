#!/bin/bash

BASE_DIR=$(pwd)

# Remove previous build artifacts
rm -rf ${BASE_DIR}/build
rm -rf ${BASE_DIR}/dist
rm -rf ${BASE_DIR}/*.egg-info

# Build and install the package
echo "Building test_npu_examples..."
python3 setup.py build bdist_wheel

if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed!"
    exit 1
fi

echo "[INFO] Build success!"

cd ${BASE_DIR}/dist
pip3 install --force-reinstall *.whl

if [ $? -ne 0 ]; then
    echo "[ERROR] Install failed!"
    exit 1
fi

echo "[INFO] Install success!"

# Run tests with pytest
cd ${BASE_DIR}
echo "Running tests..."
pytest -v test_npu_examples.py || {
    echo "[ERROR]: Pytest failed";
    exit 1;
}
echo "[INFO]: Pytest success!"
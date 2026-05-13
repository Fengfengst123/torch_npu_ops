/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#pragma once

#include <acl/acl.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

#include "args_builder.h"
#include "kernel_registry.h"

namespace xllm::kernel::npu {

inline bool is_regular_file_path(const std::filesystem::path& path) {
  std::error_code error_code;
  return std::filesystem::is_regular_file(path, error_code);
}

inline void append_unique_path(std::vector<std::filesystem::path>* paths,
                               const std::filesystem::path& path) {
  if (path.empty()) {
    return;
  }

  std::filesystem::path normalized_path = path.lexically_normal();
  for (const auto& existing_path : *paths) {
    if (existing_path == normalized_path) {
      return;
    }
  }
  paths->push_back(std::move(normalized_path));
}

inline void append_env_binary_root(std::vector<std::filesystem::path>* paths,
                                   const char* env_name) {
  const char* env_value = std::getenv(env_name);
  if (env_value == nullptr || env_value[0] == '\0') {
    return;
  }
  append_unique_path(paths, std::filesystem::path(env_value));
}

inline std::filesystem::path get_current_object_dir() {
  Dl_info dl_info{};
  if (dladdr(reinterpret_cast<const void*>(&get_current_object_dir),
             &dl_info) == 0 ||
      dl_info.dli_fname == nullptr) {
    return {};
  }

  return std::filesystem::path(dl_info.dli_fname).parent_path();
}

inline std::filesystem::path get_executable_dir() {
  std::vector<char> buffer(/*capacity=*/4096, '\0');
  ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length <= 0) {
    return {};
  }

  return std::filesystem::path(std::string(buffer.data(), length))
      .parent_path();
}

inline std::vector<std::filesystem::path> get_candidate_binary_roots() {
  std::vector<std::filesystem::path> roots;

  append_env_binary_root(&roots, "TRITON_BINARY_PATH");

#ifdef TRITON_BINARY_PATH
  append_unique_path(&roots, std::filesystem::path(TRITON_BINARY_PATH));
#endif

  std::filesystem::path current_object_dir = get_current_object_dir();
  if (!current_object_dir.empty()) {
    append_unique_path(&roots, current_object_dir / "triton_npu" / "binary");
  }

  std::filesystem::path executable_dir = get_executable_dir();
  if (!executable_dir.empty()) {
    append_unique_path(&roots, executable_dir / "triton_npu" / "binary");
  }

  return roots;
}

class OperationBase {
 public:
  explicit OperationBase(std::string kernel_name, std::string npubin_path = "")
      : kernel_name_(std::move(kernel_name)),
        npubin_path_(std::move(npubin_path)) {}

  virtual ~OperationBase() {
    std::lock_guard<std::mutex> guard(pending_mu_);
    for (auto& pending : pending_releases_) {
      if (pending.event != nullptr) {
        aclrtSynchronizeEvent(pending.event);
        aclrtDestroyEvent(pending.event);
      }
      if (pending.workspace) {
        aclrtFree(pending.workspace);
      }
      if (pending.lock) {
        aclrtFree(pending.lock);
      }
    }
    pending_releases_.clear();
  }

  template <class BuildArgsFn>
  rtError_t execute(rtStream_t stream,
                    int32_t gridX,
                    int32_t gridY,
                    int32_t gridZ,
                    BuildArgsFn&& build_args) {
    aclmdlRICaptureStatus capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI model_ri = nullptr;
    (void)aclmdlRICaptureGetInfo(stream, &capture_status, &model_ri);
    if (!ensure_registered()) {
      return static_cast<rtError_t>(-1);
    }

    const uint32_t block_num = static_cast<uint32_t>(gridX) *
                               static_cast<uint32_t>(gridY) *
                               static_cast<uint32_t>(gridZ);

    void* ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    auto rt_ret =
        rtGetC2cCtrlAddr(reinterpret_cast<uint64_t*>(&ffts_addr), &ffts_len);
    if (rt_ret != RT_ERROR_NONE) {
      LOG(ERROR) << "rtGetC2cCtrlAddr failed: " << rt_ret;
      return rt_ret;
    }

    void* workspace = nullptr;
    void* lock = nullptr;
    const auto acl_ret = setup_workspace(block_num, &workspace, &lock);
    if (acl_ret != ACL_ERROR_NONE) {
      return static_cast<rtError_t>(acl_ret);
    }

    ArgsBuilder ab;
    ab.add_aligned<void*>(ffts_addr, 8);
    ab.add_aligned<void*>(lock, 8);
    ab.add_aligned<void*>(workspace, 8);
    build_args(ab);
    ab.add_aligned<int32_t>(gridX, 4);
    ab.add_aligned<int32_t>(gridY, 4);
    ab.add_aligned<int32_t>(gridZ, 4);

    KernelStubHandle stub =
        KernelRegistry::get_instance().get_kernel_stub(kernel_name_);
    if (stub == nullptr) {
      LOG(ERROR) << "Kernel stub is null for '" << kernel_name_ << "'";
      cleanup_workspace(workspace, lock);
      return static_cast<rtError_t>(-1);
    }

    rt_ret = rtKernelLaunch(stub,
                            block_num,
                            const_cast<void*>(ab.data()),
                            static_cast<uint32_t>(ab.size()),
                            nullptr,
                            stream);

    if (capture_status != ACL_MODEL_RI_CAPTURE_STATUS_NONE) {
      cleanup_workspace(workspace, lock);
      cleanup_completed_releases();
      return rt_ret;
    }

    // In graph capture mode, aclrtRecordEvent is not supported (207000).
    // We skip async release and free workspace immediately, because graph
    // capture only records operations; the actual execution happens later
    // when the graph is replayed, and the graph mempool manages temporary
    // allocations independently.
    // In eager mode, we use an event to release workspace asynchronously
    // without blocking the host.
    aclrtEvent event = nullptr;
    auto event_ret = aclrtCreateEvent(&event);
    if (event_ret == ACL_ERROR_NONE) {
      event_ret = aclrtRecordEvent(event, stream);
      if (event_ret == ACL_ERROR_NONE) {
        std::lock_guard<std::mutex> guard(pending_mu_);
        pending_releases_.push_back({workspace, lock, event});
      } else if (event_ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        // Graph capture mode: event recording is unsupported.
        // Destroy the created event and free workspace synchronously.
        aclrtDestroyEvent(event);
        cleanup_workspace(workspace, lock);
      } else {
        LOG(WARNING) << "aclrtRecordEvent failed for '" << kernel_name_
                     << "': " << event_ret;
        aclrtDestroyEvent(event);
        cleanup_workspace(workspace, lock);
      }
    } else {
      LOG(WARNING) << "aclrtCreateEvent failed for '" << kernel_name_
                   << "': " << event_ret;
      cleanup_workspace(workspace, lock);
    }

    // Opportunistically clean up completed releases from previous calls.
    cleanup_completed_releases();

    return rt_ret;
  }

 protected:
  const std::string& kernel_name() const { return kernel_name_; }

  virtual std::string resolve_npubin_path() const {
    if (!npubin_path_.empty()) {
      return npubin_path_;
    }

    std::string kernel_file_name = kernel_name_ + ".npubin";
    for (const auto& binary_root : get_candidate_binary_roots()) {
      std::filesystem::path candidate_path = binary_root / kernel_file_name;
      if (is_regular_file_path(candidate_path)) {
        return candidate_path.string();
      }
    }

#ifdef TRITON_BINARY_PATH
    return (std::filesystem::path(TRITON_BINARY_PATH) / kernel_file_name)
        .string();
#else
    return {};
#endif
  }

  bool ensure_registered() {
    auto& reg = KernelRegistry::get_instance();
    if (reg.is_kernel_registered(kernel_name_)) {
      return true;
    }

    const std::string bin = resolve_npubin_path();
    if (bin.empty()) {
      LOG(ERROR) << "Empty npubin path for kernel '" << kernel_name_ << "'";
      return false;
    }
    if (!reg.register_kernel(kernel_name_, bin)) {
      LOG(ERROR) << "Failed to register kernel '" << kernel_name_ << "' from "
                 << bin;
      return false;
    }
    return true;
  }

  aclError setup_workspace(uint32_t block_num, void** workspace, void** lock) {
    *workspace = nullptr;
    *lock = nullptr;

    int64_t workspace_size = -1;
    int64_t lock_init_value = 0;
    int64_t lock_num = -1;

    auto& reg = KernelRegistry::get_instance();
    reg.get_kernel_workspace_config(
        kernel_name_, workspace_size, lock_init_value, lock_num);

    if (workspace_size > 0) {
      workspace_size *= static_cast<int64_t>(block_num);
      const auto ret =
          aclrtMalloc(workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtMalloc workspace failed for '" << kernel_name_
                   << "': " << ret;
        return ret;
      }
    }

    if (lock_num > 0) {
      const uint64_t bytes = static_cast<uint64_t>(lock_num) * sizeof(int64_t);
      auto ret = aclrtMalloc(lock, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtMalloc lock failed for '" << kernel_name_
                   << "': " << ret;
        if (*workspace) {
          aclrtFree(*workspace);
          *workspace = nullptr;
        }
        return ret;
      }

      std::vector<int64_t> init(static_cast<size_t>(lock_num), lock_init_value);
      ret = aclrtMemcpy(
          *lock, bytes, init.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtMemcpy lock init failed for '" << kernel_name_
                   << "': " << ret;
        if (*workspace) {
          aclrtFree(*workspace);
          *workspace = nullptr;
        }
        if (*lock) {
          aclrtFree(*lock);
          *lock = nullptr;
        }
        return ret;
      }
    }

    return ACL_ERROR_NONE;
  }

  void cleanup_workspace(void* workspace, void* lock) {
    if (workspace) {
      aclrtFree(workspace);
    }
    if (lock) {
      aclrtFree(lock);
    }
  }

  struct PendingRelease {
    void* workspace = nullptr;
    void* lock = nullptr;
    aclrtEvent event = nullptr;
  };

  void cleanup_completed_releases() {
    std::lock_guard<std::mutex> guard(pending_mu_);
    auto it = pending_releases_.begin();
    while (it != pending_releases_.end()) {
      if (it->event == nullptr) {
        // Already freed or fallback path
        it = pending_releases_.erase(it);
        continue;
      }

      aclrtEventStatus status = ACL_EVENT_STATUS_NOT_READY;
      auto ret = aclrtQueryEvent(it->event, &status);
      if (ret != ACL_ERROR_NONE) {
        LOG(WARNING) << "aclrtQueryEvent failed for '" << kernel_name_
                     << "': " << ret;
        ++it;
        continue;
      }

      if (status == ACL_EVENT_STATUS_COMPLETE) {
        if (it->workspace) {
          aclrtFree(it->workspace);
        }
        if (it->lock) {
          aclrtFree(it->lock);
        }
        aclrtDestroyEvent(it->event);
        it = pending_releases_.erase(it);
      } else {
        ++it;
      }
    }
  }

 private:
  std::string kernel_name_;
  std::string npubin_path_;
  std::mutex pending_mu_;
  std::vector<PendingRelease> pending_releases_;
};

}  // namespace xllm::kernel::npu

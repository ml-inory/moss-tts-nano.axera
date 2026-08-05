#include "ax_runner.h"

#include <stdexcept>

#ifdef MOSS_WITH_AX_ENGINE

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <cstring>
#include <fstream>
#include <iterator>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace {

struct MappedFile {
    int fd = -1;
    void* data = MAP_FAILED;
    size_t size = 0;

    explicit MappedFile(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("failed to open " + path);
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("failed to stat " + path);
        }
        size = static_cast<size_t>(st.st_size);
        data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("failed to mmap " + path);
        }
    }

    ~MappedFile() {
        if (data != MAP_FAILED) {
            ::munmap(data, size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }
};

size_t io_size(const AX_ENGINE_IOMETA_T& meta) {
    size_t size = 1;
    for (int d = 0; d < meta.nShapeSize; ++d) {
        size *= static_cast<size_t>(meta.pShape[d]);
    }
    switch (meta.eDataType) {
        case AX_ENGINE_DT_UINT8:
        case AX_ENGINE_DT_SINT8:
            return size;
        case AX_ENGINE_DT_UINT16:
        case AX_ENGINE_DT_SINT16:
        case AX_ENGINE_DT_BFLOAT16:
            return size * 2;
        default:
            return size * 4;
    }
}

void check_ax(int ret, const char* message) {
    if (ret != 0) {
        throw std::runtime_error(message);
    }
}

}  // namespace

struct AxRunner::Impl {
    AX_ENGINE_HANDLE handle = nullptr;
    AX_ENGINE_IO_INFO_T* info = nullptr;
    AX_ENGINE_IO_T io{};

    explicit Impl(const std::string& model_path) {
        check_ax(AX_SYS_Init(), "AX_SYS_Init failed");

        AX_ENGINE_NPU_ATTR_T npu_attr;
        std::memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
        check_ax(AX_ENGINE_Init(&npu_attr), "AX_ENGINE_Init failed");

        MappedFile model(model_path);
        check_ax(AX_ENGINE_CreateHandle(&handle, model.data, static_cast<AX_U32>(model.size)),
                 "AX_ENGINE_CreateHandle failed");
        check_ax(AX_ENGINE_CreateContext(handle), "AX_ENGINE_CreateContext failed");
        check_ax(AX_ENGINE_GetIOInfo(handle, &info), "AX_ENGINE_GetIOInfo failed");

        io.nInputSize = info->nInputSize;
        io.nOutputSize = info->nOutputSize;
        io.pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize]();
        io.pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize]();
        for (uint32_t i = 0; i < info->nInputSize; ++i) {
            auto& buf = io.pInputs[i];
            buf.nSize = info->pInputs[i].nSize;
            check_ax(AX_SYS_MemAlloc(&buf.phyAddr, &buf.pVirAddr, buf.nSize, 128,
                                     reinterpret_cast<const AX_S8*>(info->pInputs[i].pName)),
                     "AX_SYS_MemAlloc input");
        }
        for (uint32_t i = 0; i < info->nOutputSize; ++i) {
            auto& buf = io.pOutputs[i];
            buf.nSize = info->pOutputs[i].nSize;
            check_ax(AX_SYS_MemAlloc(&buf.phyAddr, &buf.pVirAddr, buf.nSize, 128,
                                     reinterpret_cast<const AX_S8*>(info->pOutputs[i].pName)),
                     "AX_SYS_MemAlloc output");
        }
    }

    ~Impl() {
        if (io.pInputs != nullptr) {
            for (uint32_t i = 0; i < info->nInputSize; ++i) {
                if (io.pInputs[i].phyAddr != 0) {
                    AX_SYS_MemFree(io.pInputs[i].phyAddr, io.pInputs[i].pVirAddr);
                }
            }
            delete[] io.pInputs;
        }
        if (io.pOutputs != nullptr) {
            for (uint32_t i = 0; i < info->nOutputSize; ++i) {
                if (io.pOutputs[i].phyAddr != 0) {
                    AX_SYS_MemFree(io.pOutputs[i].phyAddr, io.pOutputs[i].pVirAddr);
                }
            }
            delete[] io.pOutputs;
        }
        if (handle != nullptr) {
            AX_ENGINE_DestroyHandle(handle);
        }
        AX_ENGINE_Deinit();
        AX_SYS_Deinit();
    }
};

AxRunner::AxRunner(const std::string& model_path) : impl_(new Impl(model_path)) {}
AxRunner::~AxRunner() { delete impl_; }

std::vector<size_t> AxRunner::input_sizes() const {
    std::vector<size_t> sizes;
    for (uint32_t i = 0; i < impl_->info->nInputSize; ++i) {
        sizes.push_back(io_size(impl_->info->pInputs[i]));
    }
    return sizes;
}

std::vector<size_t> AxRunner::output_sizes() const {
    std::vector<size_t> sizes;
    for (uint32_t i = 0; i < impl_->info->nOutputSize; ++i) {
        sizes.push_back(io_size(impl_->info->pOutputs[i]));
    }
    return sizes;
}

std::vector<std::string> AxRunner::input_names() const {
    std::vector<std::string> names;
    for (uint32_t i = 0; i < impl_->info->nInputSize; ++i) {
        names.emplace_back(impl_->info->pInputs[i].pName);
    }
    return names;
}

std::vector<std::string> AxRunner::output_names() const {
    std::vector<std::string> names;
    for (uint32_t i = 0; i < impl_->info->nOutputSize; ++i) {
        names.emplace_back(impl_->info->pOutputs[i].pName);
    }
    return names;
}

std::vector<std::vector<uint8_t>> AxRunner::run(
    const std::vector<std::pair<const void*, size_t>>& feeds) {
    if (feeds.size() != impl_->info->nInputSize) {
        throw std::runtime_error("input count mismatch");
    }
    for (uint32_t i = 0; i < impl_->info->nInputSize; ++i) {
        size_t want = input_sizes()[i];
        if (feeds[i].second != want) {
            throw std::runtime_error("input size mismatch for tensor " + std::to_string(i));
        }
        std::memcpy(impl_->io.pInputs[i].pVirAddr, feeds[i].first, feeds[i].second);
    }
    check_ax(AX_ENGINE_RunSync(impl_->handle, &impl_->io), "AX_ENGINE_RunSync failed");

    std::vector<std::vector<uint8_t>> outputs;
    for (uint32_t i = 0; i < impl_->info->nOutputSize; ++i) {
        size_t size = output_sizes()[i];
        const uint8_t* src = static_cast<const uint8_t*>(impl_->io.pOutputs[i].pVirAddr);
        outputs.emplace_back(static_cast<const uint8_t*>(src),
                             static_cast<const uint8_t*>(src) + size);
    }
    return outputs;
}

#else  // MOSS_WITH_AX_ENGINE

struct AxRunner::Impl {
    std::string model_path;
    explicit Impl(const std::string& path) : model_path(path) {}
};

AxRunner::AxRunner(const std::string& model_path) : impl_(new Impl(model_path)) {
    throw std::runtime_error(
        "AxRunner compiled without AX runtime; configure with -DAX_RUNTIME_ROOT=<bsp root>");
}
AxRunner::~AxRunner() { delete impl_; }
std::vector<size_t> AxRunner::input_sizes() const { return {}; }
std::vector<size_t> AxRunner::output_sizes() const { return {}; }
std::vector<std::string> AxRunner::input_names() const { return {}; }
std::vector<std::string> AxRunner::output_names() const { return {}; }
std::vector<std::vector<uint8_t>> AxRunner::run(
    const std::vector<std::pair<const void*, size_t>>&) {
    throw std::runtime_error("AxRunner stub: no AX runtime");
}

#endif

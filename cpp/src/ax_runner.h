#pragma once

// Thin wrapper over the AX Engine runtime (ax_engine/ax_sys from the AXera
// BSP, AX650 convention). Compiled against the real runtime when CMake is
// configured with -DAX_RUNTIME_ROOT=<bsp root>; otherwise a stub that throws
// on construction (host-side configure/build still passes — see README).

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

class AxRunner {
public:
    explicit AxRunner(const std::string& model_path);
    ~AxRunner();

    AxRunner(const AxRunner&) = delete;
    AxRunner& operator=(const AxRunner&) = delete;

    std::vector<size_t> input_sizes() const;
    std::vector<size_t> output_sizes() const;
    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;

    // feeds[i] = (data, bytes) for input i; returns output byte buffers in
    // model output order.
    std::vector<std::vector<uint8_t>> run(
        const std::vector<std::pair<const void*, size_t>>& feeds);

private:
    struct Impl;
    Impl* impl_;
};

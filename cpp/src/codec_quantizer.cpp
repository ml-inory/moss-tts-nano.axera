#include "codec_quantizer.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace moss {

namespace {

std::vector<float> load_floats(std::ifstream& file, size_t count) {
    std::vector<float> data(count);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(count * sizeof(float)));
    if (!file) {
        throw std::runtime_error("codec_quantizer_cpp.bin truncated");
    }
    return data;
}

}  // namespace

CodecQuantizer::CodecQuantizer(const std::string& blob_path) {
    std::ifstream file(blob_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("failed to open " + blob_path);
    }
    codebooks_ = load_floats(file, kNumCodebooks * kCodebookSize * kCodeDim);
    out_proj_w_ = load_floats(file, kNumCodebooks * kCodeDim * kRvqDim);
    out_proj_b_ = load_floats(file, kNumCodebooks * kRvqDim);
    output_proj_w_ = load_floats(file, kRvqDim * kHiddenDim);
    output_proj_b_ = load_floats(file, kHiddenDim);
}

void CodecQuantizer::codes_to_emb(
    const std::vector<std::vector<int64_t>>& codes,
    std::vector<std::vector<float>>& codes_emb_out) const {
    if (codes.size() != kNumCodebooks) {
        throw std::runtime_error("codes must have 16 codebooks");
    }
    const size_t t = codes[0].size();
    for (int i = 1; i < kNumCodebooks; ++i) {
        if (codes[static_cast<size_t>(i)].size() != t) {
            throw std::runtime_error("codebook lengths differ");
        }
    }
    codes_emb_out.assign(kRvqDim, std::vector<float>(t, 0.0f));
    for (int cb = 0; cb < kNumCodebooks; ++cb) {
        const float* book = codebooks_.data() + cb * kCodebookSize * kCodeDim;
        const float* w = out_proj_w_.data() + cb * kCodeDim * kRvqDim;
        const float* b = out_proj_b_.data() + cb * kRvqDim;
        for (size_t ti = 0; ti < t; ++ti) {
            const int64_t idx = codes[static_cast<size_t>(cb)][ti];
            if (idx < 0 || idx >= kCodebookSize) {
                throw std::runtime_error("code index out of range");
            }
            const float* vec = book + static_cast<size_t>(idx) * kCodeDim;
            for (int j = 0; j < kRvqDim; ++j) {
                float acc = b[j];
                for (int d = 0; d < kCodeDim; ++d) {
                    acc += vec[d] * w[d * kRvqDim + j];
                }
                codes_emb_out[static_cast<size_t>(j)][ti] += acc;
            }
        }
    }
}

}  // namespace moss

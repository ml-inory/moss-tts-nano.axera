#pragma once

#include <cstdint>
#include <string>
#include <vector>

// MOSS-Audio-Tokenizer-Nano 查表权重 (CPU 侧)。
//
// codec_quantizer_cpp.bin 布局 (小端 fp32):
//   codebooks[16][1024][8]
//   out_proj_w[16][8][512]   (out = vec @ w^T, 即 w 的列是输出通道)
//   out_proj_b[16][512]
//   output_proj_w[512][768]
//   output_proj_b[768]
namespace moss {

constexpr int kNumCodebooks = 16;
constexpr int kCodebookSize = 1024;
constexpr int kCodeDim = 8;
constexpr int kRvqDim = 512;
constexpr int kHiddenDim = 768;

class CodecQuantizer {
public:
    // 从二进制文件加载权重。
    explicit CodecQuantizer(const std::string& blob_path);

    // codes: [16][T] int64 -> codes_emb: [512][T] float32
    // 与 Python 端 codes_to_emb 完全一致。
    void codes_to_emb(const std::vector<std::vector<int64_t>>& codes,
                      std::vector<std::vector<float>>& codes_emb_out) const;

    int codebook_size() const { return kCodebookSize; }

private:
    std::vector<float> codebooks_;      // [16][1024][8]
    std::vector<float> out_proj_w_;     // [16][8][512]
    std::vector<float> out_proj_b_;     // [16][512]
    std::vector<float> output_proj_w_;  // [512][768]
    std::vector<float> output_proj_b_;  // [768]
};

}  // namespace moss

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ax_runner.h"

namespace moss {

// codec_decoder.axmodel 封装: codes_emb [1,512,64] -> waveform [1,2,245760]
class CodecDecoder {
public:
    explicit CodecDecoder(const std::string& axmodel_path);

    // 输入 [512*64] float32 (NCHW/BCT 展平), 输出 [2*245760] float32。
    std::vector<float> decode(const std::vector<float>& codes_emb);

private:
    AxRunner runner_;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
};

}  // namespace moss

#include "codec_decode.h"

#include <cstring>
#include <stdexcept>

namespace moss {

CodecDecoder::CodecDecoder(const std::string& axmodel_path) : runner_(axmodel_path) {
    const auto in_sizes = runner_.input_sizes();
    const auto out_sizes = runner_.output_sizes();
    if (in_sizes.size() != 1 || out_sizes.size() != 1) {
        throw std::runtime_error("codec decoder expects 1 input and 1 output");
    }
    input_bytes_ = in_sizes[0];
    output_bytes_ = out_sizes[0];
}

std::vector<float> CodecDecoder::decode(const std::vector<float>& codes_emb) {
    if (codes_emb.size() * sizeof(float) != input_bytes_) {
        throw std::runtime_error("codes_emb size mismatch");
    }
    auto outputs = runner_.run({{codes_emb.data(), input_bytes_}});
    if (outputs[0].size() != output_bytes_) {
        throw std::runtime_error("output size mismatch");
    }
    std::vector<float> waveform(output_bytes_ / sizeof(float));
    std::memcpy(waveform.data(), outputs[0].data(), output_bytes_);
    return waveform;
}

}  // namespace moss

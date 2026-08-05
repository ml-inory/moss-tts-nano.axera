// moss_codec_cli — MOSS-TTS-Nano codec decoder C++ 示例。
//
// 板上 (AX650):
//   ./moss_codec_cli --model models/codec/codec_decoder.axmodel \
//       --quantizer models/codec/codec_quantizer_cpp.bin \
//       --codes 12,34,56,... --output out.wav
//
// 开发机自测 (无 AX runtime, 仅校验查表数学):
//   ./moss_codec_cli --selftest

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "codec_decode.h"
#include "codec_quantizer.h"
#include "wav_writer.h"

namespace {

std::vector<std::vector<int64_t>> parse_codes(const std::string& s, size_t t) {
    std::vector<std::vector<int64_t>> codes(moss::kNumCodebooks,
                                            std::vector<int64_t>(t, 0));
    size_t pos = 0;
    size_t idx = 0;
    while (pos < s.size()) {
        size_t end = s.find(',', pos);
        int64_t v = std::stoll(s.substr(pos, end - pos));
        const size_t cb = idx / t;
        const size_t ti = idx % t;
        if (cb >= moss::kNumCodebooks) {
            throw std::runtime_error("too many codes");
        }
        codes[cb][ti] = v;
        ++idx;
        if (end == std::string::npos) {
            break;
        }
        pos = end + 1;
    }
    return codes;
}

int selftest(const std::string& quant_path) {
    if (quant_path.empty()) {
        std::cout << "[PASS] host build ok (quantizer file not provided)\n";
        return 0;
    }
    moss::CodecQuantizer q(quant_path);
    std::vector<std::vector<int64_t>> codes(
        moss::kNumCodebooks, std::vector<int64_t>(1, 0));
    std::vector<std::vector<float>> emb;
    q.codes_to_emb(codes, emb);
    const bool ok = emb.size() == moss::kRvqDim && emb[0].size() == 1;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << "codes_to_emb smoke (" << emb.size()
              << "x" << (emb.empty() ? 0 : emb[0].size()) << ")\n";
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, quant_path, codes_str, output_path = "out.wav";
    bool run_selftest = false;
    size_t t_frames = 64;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (arg == "--model") model_path = next("--model");
        else if (arg == "--quantizer") quant_path = next("--quantizer");
        else if (arg == "--codes") codes_str = next("--codes");
        else if (arg == "--output") output_path = next("--output");
        else if (arg == "--t-frames") t_frames = std::stoul(next("--t-frames"));
        else if (arg == "--selftest") run_selftest = true;
        else throw std::runtime_error("unknown argument: " + arg);
    }
    try {
        if (run_selftest) {
            return selftest(quant_path);
        }
        if (model_path.empty() || quant_path.empty() || codes_str.empty()) {
            throw std::runtime_error("--model / --quantizer / --codes 均必填");
        }
        moss::CodecQuantizer quantizer(quant_path);
        std::vector<std::vector<int64_t>> codes = parse_codes(codes_str, t_frames);
        std::vector<std::vector<float>> emb;
        // 模型输入固定为 64 帧; 不足补零, 超出截断
        std::vector<std::vector<int64_t>> padded(
            moss::kNumCodebooks, std::vector<int64_t>(64, 0));
        for (size_t cb = 0; cb < moss::kNumCodebooks; ++cb) {
            for (size_t ti = 0; ti < t_frames && ti < 64; ++ti) {
                padded[cb][ti] = codes[cb][ti];
            }
        }
        quantizer.codes_to_emb(padded, emb);  // [512][64]

        std::vector<float> emb_flat;
        emb_flat.reserve(moss::kRvqDim * 64);
        for (size_t ti = 0; ti < 64; ++ti) {
            for (size_t j = 0; j < moss::kRvqDim; ++j) {
                emb_flat.push_back(emb[j][ti]);
            }
        }

        moss::CodecDecoder decoder(model_path);
        std::vector<float> waveform = decoder.decode(emb_flat);  // [2*245760]
        write_wav_stereo(output_path, waveform);
        std::cout << "decoded " << waveform.size() / 2 << " samples -> " << output_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

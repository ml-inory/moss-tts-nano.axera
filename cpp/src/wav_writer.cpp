#include "wav_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

void write_le32(std::ofstream& f, uint32_t v) {
    char b[4] = {static_cast<char>(v & 0xff), static_cast<char>((v >> 8) & 0xff),
                 static_cast<char>((v >> 16) & 0xff), static_cast<char>((v >> 24) & 0xff)};
    f.write(b, 4);
}

void write_le16(std::ofstream& f, uint16_t v) {
    char b[2] = {static_cast<char>(v & 0xff), static_cast<char>((v >> 8) & 0xff)};
    f.write(b, 2);
}

}  // namespace

void write_wav_stereo(const std::string& path, const std::vector<float>& waveform,
                      int sample_rate) {
    if (waveform.empty()) {
        throw std::runtime_error("empty waveform");
    }
    const uint32_t samples = static_cast<uint32_t>(waveform.size());
    const uint32_t data_bytes = samples * sizeof(int16_t);
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open " + path);
    }
    f.write("RIFF", 4);
    write_le32(f, 36 + data_bytes);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    write_le32(f, 16);
    write_le16(f, 1);
    write_le16(f, 2);  // 双声道
    write_le32(f, static_cast<uint32_t>(sample_rate));
    write_le32(f, static_cast<uint32_t>(sample_rate) * 2 * 2);
    write_le16(f, 4);
    write_le16(f, 16);
    f.write("data", 4);
    write_le32(f, data_bytes);
    for (float v : waveform) {
        v = std::max(-1.0f, std::min(1.0f, v));
        write_le16(f, static_cast<uint16_t>(static_cast<int16_t>(std::round(v * 32767.0f))));
    }
}

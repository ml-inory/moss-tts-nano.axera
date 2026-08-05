#pragma once

#include <string>
#include <vector>

// 写 48kHz 双声道 16-bit PCM WAV。waveform 布局为 [2][N] 展平。
void write_wav_stereo(const std::string& path, const std::vector<float>& waveform,
                      int sample_rate = 48000);

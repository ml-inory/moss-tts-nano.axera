cmake_minimum_required(VERSION 3.16)
project(moss_tts_codec CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# AXera BSP runtime root (contains include/ with ax_engine_api.h, ax_sys_api.h
# and lib/ with libax_engine.so, libax_sys.so). Leave empty for a host
# configure/build check — the engine wrapper compiles to a throwing stub.
set(AX_RUNTIME_ROOT "" CACHE PATH "AX BSP runtime root (include/ + lib/)")

add_library(moss_codec STATIC
  src/ax_runner.cpp
  src/codec_quantizer.cpp
  src/codec_decode.cpp
  src/wav_writer.cpp
)
target_include_directories(moss_codec PUBLIC src)

if(AX_RUNTIME_ROOT)
  target_compile_definitions(moss_codec PRIVATE MOSS_WITH_AX_ENGINE=1)
  target_include_directories(moss_codec PRIVATE ${AX_RUNTIME_ROOT}/include)
  target_link_directories(moss_codec PUBLIC ${AX_RUNTIME_ROOT}/lib)
  target_link_libraries(moss_codec PUBLIC ax_engine ax_sys ax_interpreter)
  message(STATUS "moss_codec: AX runtime from ${AX_RUNTIME_ROOT}")
else()
  message(STATUS "moss_codec: AX_RUNTIME_ROOT not set — building host stub (no NPU runtime)")
endif()

add_executable(moss_codec_cli src/main.cpp)
target_link_libraries(moss_codec_cli PRIVATE moss_codec)

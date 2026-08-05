#!/usr/bin/env bash
set -e

echo "Pulsar2 编译: codec_decoder.onnx -> model.axmodel (AX650 / NPU3)"
echo "镜像: pulsar2:7.0-lite (由本机 ax_pulsar2_7.0_lite.tar.gz 导入)"

IMAGE=${PULSAR2_IMAGE:-pulsar2:7.0-lite}
WORKSPACE=${WORKSPACE:-$(pwd)/../..}

docker load -i "$WORKSPACE/ax_pulsar2_7.0_lite.tar.gz"

docker run --rm --network host \
  -v "$WORKSPACE:/workspace" \
  -v /var/hasplm:/var/hasplm \
  -v /tmp/p2_verify_home/.hasplm:/root/.hasplm \
  -e HASP_HOME=/root/.hasplm \
  "$IMAGE" -lc "set +e; PATH=/usr/local/bin/.venv/bin:/opt/pulsar2:\$PATH \
  pulsar2 build --config /workspace/package/model_convert/pulsar2_config.json; \
  status=\$?; chown -R $(id -u):$(id -g) /workspace; exit \$status"

echo "编译产物: $WORKSPACE/package/model_convert/model.axmodel"

#!/usr/bin/env bash
# Side-load an Ollama model into a Docker volume without running `ollama pull` inside the container.
# Needed on hosts whose egress proxy breaks TLS for containers. Fetches manifest + blobs with host curl,
# verifies sha256, and copies them into the volume layout Ollama expects.
# Usage: scripts/ollama_sideload.sh <model> <tag> [volume]   e.g. scripts/ollama_sideload.sh qwen2.5 7b sigint_ollama
set -euo pipefail
MODEL=${1:?model}; TAG=${2:?tag}; VOL=${3:-sigint_ollama}
REG=https://registry.ollama.ai/v2/library/$MODEL
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
echo "fetching manifest $MODEL:$TAG"
curl -fsSL -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' "$REG/manifests/$TAG" -o "$WORK/manifest.json"
DIGESTS=$(python3 -c "import json;m=json.load(open('$WORK/manifest.json'));print(m['config']['digest']);[print(l['digest']) for l in m['layers']]")
mkdir -p "$WORK/blobs"
for d in $DIGESTS; do
  hex=${d#sha256:}; out="$WORK/blobs/sha256-$hex"
  echo "blob $hex"; curl -fSL --retry 3 "$REG/blobs/$d" -o "$out"
  echo "$hex  $out" | sha256sum -c --quiet
done
docker volume create "$VOL" >/dev/null
docker run --rm -v "$VOL:/root/.ollama" -v "$WORK:/w" python:3.12-slim sh -c "
  mkdir -p /root/.ollama/models/manifests/registry.ollama.ai/library/$MODEL /root/.ollama/models/blobs &&
  cp /w/manifest.json /root/.ollama/models/manifests/registry.ollama.ai/library/$MODEL/$TAG &&
  cp /w/blobs/* /root/.ollama/models/blobs/ && ls /root/.ollama/models/manifests/registry.ollama.ai/library/"
echo "done: $MODEL:$TAG in volume $VOL"

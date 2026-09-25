# isa-agent: every target is reproducible on a Linux host with Docker (+ an NVIDIA GPU for the LLM runs).
PY ?= .venv/bin/python
MODEL ?= qwen3-coder:30b

.PHONY: venv image isa baselines test lint ollama models eval mlir report all

venv:
	uv venv -p 3.11 .venv && uv pip install -p $(PY) -e ".[dev]"

image:                      ## clang 18, MLIR 18, QEMU 8.2 (riscv64), riscv64 sysroot
	docker build -t isa-agent-tools:latest docker/

isa:                        ## regenerate isa/include/xdsp.h and docs/XDSP_MANUAL.md from isaagent/isa_spec.py
	$(PY) -m isaagent.gen_isa

baselines: isa              ## scalar / clang autovec / hand-written instruction counts
	$(PY) -m isaagent.baselines

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check isaagent scripts tests

ollama:                     ## local model server, bound to 127.0.0.1 only
	docker run -d --name isa-ollama --gpus all -v isa_ollama:/root/.ollama -p 127.0.0.1:11435:11434 \
	  -e OLLAMA_NUM_PARALLEL=4 -e OLLAMA_FLASH_ATTENTION=1 -e OLLAMA_KEEP_ALIVE=2h ollama/ollama:latest

models:                     ## side-load the two models (works behind TLS-intercepting proxies)
	scripts/ollama_sideload.sh qwen2.5-coder 7b isa_ollama
	scripts/ollama_sideload.sh qwen3-coder 30b isa_ollama

eval:                       ## resumable; appends to results/episodes.jsonl
	$(PY) -m isaagent.run_eval --model $(MODEL) --seeds 3 --workers 4

mlir:                       ## ONNX -> MLIR -> LLVM -> RV64GCV on QEMU, vs C and hand-written kernels
	$(PY) -m isaagent.mlir_flow

report:
	$(PY) -m isaagent.report

all: image isa baselines test mlir eval report

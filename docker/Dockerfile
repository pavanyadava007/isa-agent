# Toolchain image: LLVM/clang 18, MLIR 18 tools, QEMU user-mode RISC-V, riscv64 glibc sysroot.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      clang-18 lld-18 llvm-18 mlir-18-tools libmlir-18-dev \
      qemu-user gcc-riscv64-linux-gnu libc6-dev-riscv64-cross \
      python3 python3-numpy make ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN for t in clang clang++ llc opt llvm-mc llvm-objdump llvm-mca llvm-nm llvm-readelf mlir-opt mlir-translate mlir-runner ld.lld; do \
      ln -sf /usr/bin/$t-18 /usr/local/bin/$t 2>/dev/null || ln -sf /usr/lib/llvm-18/bin/$t /usr/local/bin/$t; done
WORKDIR /work

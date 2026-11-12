# syntax=docker/dockerfile:1.7
#
# Smriti — build / test / run in containers. See docker-compose.yml for the
# day-to-day commands; the short version:
#
#   docker compose run --rm assets     # one-time: models + GeoNames into ./models ./data
#   docker compose run --rm ci         # fmt + clippy + tests + frontend checks
#   docker compose run --rm app        # build + launch the app, heavy image (GPU, CPU fallback)
#   docker compose run --rm app-cpu    # same, light image (CPU only)
#
# Build flag
#   GPU=true   (default) heavy image: CUDA 12 + cuDNN 9 base and the GPU build of
#              ONNX Runtime. Uses an NVIDIA GPU when one is available, else CPU.
#   GPU=false  light image: plain Ubuntu base and the CPU build of ONNX Runtime.
#   e.g. docker build --target runtime --build-arg GPU=false -t smriti-app:cpu .
#
# Stages (only `runtime` is shipped; everything else is build-time and discarded)
#   dev        toolchain: Rust + Node + Tauri system deps. Source is bind-mounted by compose.
#   frontend   Vite build of src-ui.
#   builder    release binary with the frontend embedded.
#   ort-gpu / ort-cpu   ONNX Runtime libraries.
#   runtime    binary + runtime libs + ONNX Runtime, on a CUDA or plain Ubuntu base.

ARG UBUNTU_VERSION=24.04
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
ARG ORT_VERSION=1.23.0
# "true" (heavy, GPU) or "false" (light, CPU only). Selects runtime-base-* and ort-*.
ARG GPU=true

# ---------------------------------------------------------------------------
# dev: everything needed to build, test and lint
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS dev

ARG NODE_VERSION=20.19.0
ARG RUST_VERSION=stable

ENV DEBIAN_FRONTEND=noninteractive

# Same package set as docs/BUILD.md and .github/workflows/ci.yml, plus a C
# toolchain and the tools scripts/setup_assets.sh needs (curl, unzip, xz).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config git curl ca-certificates unzip xz-utils patchelf \
        libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev \
        librsvg2-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev \
        libdbus-1-dev libheif-dev \
    && rm -rf /var/lib/apt/lists/*

# Node (the frontend is Vite 8 + Svelte 5, which needs 20.19+).
RUN curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
    | tar -xJ -C /usr/local --strip-components=1

# Rust. Toolchain lives in /usr/local so any uid can use it; per-user state
# (registry, build output, npm cache) goes to /cache, a world-writable volume.
ENV RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:$PATH
RUN CARGO_HOME=/usr/local/cargo curl -fsSL https://sh.rustup.rs \
    | CARGO_HOME=/usr/local/cargo sh -s -- -y --profile minimal \
        --default-toolchain "${RUST_VERSION}" -c clippy -c rustfmt \
    && chmod -R a+rX /usr/local/rustup /usr/local/cargo

RUN mkdir -p /cache /workspace && chmod 1777 /cache /workspace
ENV CARGO_HOME=/cache/cargo \
    CARGO_TARGET_DIR=/cache/target \
    npm_config_cache=/cache/npm \
    HOME=/tmp
WORKDIR /workspace
CMD ["bash"]

# ---------------------------------------------------------------------------
# frontend: Vite build, embedded into the binary at compile time
# ---------------------------------------------------------------------------
FROM dev AS frontend
WORKDIR /src-ui
COPY src-ui/package.json src-ui/package-lock.json ./
RUN --mount=type=cache,target=/cache/npm npm ci
COPY src-ui/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# builder: release binary (`tauri::generate_context!` needs src-ui/dist)
# ---------------------------------------------------------------------------
FROM dev AS builder
# Comma-free feature list, e.g. "heic". Empty = defaults only.
ARG CARGO_FEATURES=heic
WORKDIR /src
COPY . .
COPY --from=frontend /src-ui/dist src-ui/dist
RUN --mount=type=cache,target=/cache/cargo/registry \
    --mount=type=cache,target=/cache/cargo/git \
    --mount=type=cache,target=/cache/target \
    cargo build --release -p smriti-tauri ${CARGO_FEATURES:+--features "$CARGO_FEATURES"} \
    && mkdir -p /out && cp /cache/target/release/smriti-tauri /out/smriti-tauri

# ---------------------------------------------------------------------------
# ONNX Runtime 1.23 (matches the `ort` crate). Downloaded in a throwaway stage so
# curl and the tarball never reach the final image.
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS ort-fetch
ARG ORT_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

FROM ort-fetch AS ort-true
ARG ORT_VERSION
# CUDA 12 + cuDNN 9 build (~390 MB of libraries). Only the .so files are kept; the
# tarball's cmake/ and pkgconfig/ development files are left behind.
RUN set -eu; \
    curl -fL --retry 5 -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"; \
    mkdir -p /tmp/ort /out; \
    tar -xzf /tmp/ort.tgz -C /tmp/ort --strip-components=1; \
    cp -a /tmp/ort/lib/*.so* /out/; \
    ls /out

FROM ort-fetch AS ort-false
ARG ORT_VERSION
RUN set -eu; \
    curl -fL --retry 5 -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"; \
    mkdir -p /tmp/ort /out; \
    tar -xzf /tmp/ort.tgz -C /tmp/ort --strip-components=1; \
    cp -a /tmp/ort/lib/*.so* /out/; \
    ls /out

# Picks ort-true / ort-false from the GPU flag.
FROM ort-${GPU} AS ort

# ---------------------------------------------------------------------------
# Runtime base: heavy (CUDA + cuDNN) or light (plain Ubuntu), picked by GPU.
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS runtime-base-true
FROM ubuntu:${UBUNTU_VERSION} AS runtime-base-false

# ---------------------------------------------------------------------------
# runtime: the only stage that ships
# ---------------------------------------------------------------------------
FROM runtime-base-${GPU} AS runtime
ARG GPU
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libwebkit2gtk-4.1-0 libgtk-3-0t64 libayatana-appindicator3-1 librsvg2-2 \
        libsoup-3.0-0 libjavascriptcoregtk-4.1-0 libheif1 libdbus-1-3 \
        dbus dbus-x11 libgl1-mesa-dri libegl1 libgbm1 fonts-dejavu-core \
        shared-mime-info adwaita-icon-theme ca-certificates \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

WORKDIR /app
COPY --from=builder /out/smriti-tauri /app/smriti-tauri
# libs/onnxruntime is where the app's asset check looks (relative to CWD).
COPY --from=ort /out/ /app/libs/onnxruntime/
COPY docker/entrypoint.sh /usr/local/bin/smriti-entrypoint
RUN chmod +x /usr/local/bin/smriti-entrypoint \
    && mkdir -p /state/config/smriti /state/data/smriti /state/cache /photos \
    && chmod -R 1777 /state

LABEL org.opencontainers.image.title="Smriti" \
      in.smriti.gpu="${GPU}"

# NVIDIA_* only matter to the NVIDIA Container Toolkit and are harmless without a GPU.
ENV ORT_DYLIB_PATH=/app/libs/onnxruntime/libonnxruntime.so \
    # Settings (config/smriti) and downloaded packs (data/smriti) live under
    # /state, backed by volumes or host folders; see docker-compose.yml.
    HOME=/state \
    XDG_CONFIG_HOME=/state/config \
    XDG_DATA_HOME=/state/data \
    XDG_CACHE_HOME=/state/cache \
    GDK_BACKEND=x11 \
    # WebKit's DMABUF renderer is unreliable inside containers.
    WEBKIT_DISABLE_DMABUF_RENDERER=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    RUST_LOG=info
ENTRYPOINT ["smriti-entrypoint"]
CMD ["/app/smriti-tauri"]

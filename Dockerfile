# Multi-stage Dockerfile for arc-llama
#
# Produces an image with:
#   - Pre-built llama-server (SYCL backend)
#   - arc-llama installed from the local source tree
#   - Intel oneAPI runtime libraries for SYCL
#
# Build:
#   docker build -t arc-llama:latest .
#
# Run (requires GPU access):
#   docker run --rm -it \
#     --device /dev/dri:/dev/dri \
#     --group-add video \
#     --group-add render \
#     -p 11437:11437 \
#     -v $HOME/models:/models:ro \
#     arc-llama:latest \
#     arc-llama serve
#
# Or with a custom config:
#   docker run --rm -it ... \
#     -v $PWD/config.toml:/root/.config/arc-llama/config.toml:ro \
#     arc-llama:latest \
#     arc-llama serve

# ------------------------------------------------------------------------------
# Stage 1: Build llama-server with SYCL
# ------------------------------------------------------------------------------
FROM intel/oneapi-basekit:2025.0.0-0-devel-ubuntu22.04 AS llama-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin to a known-good llama.cpp revision (post-SYCL Battlemage fixes)
ARG LLAMA_CPP_REF=b4274
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
    https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp

WORKDIR /tmp/llama.cpp

# Build with SYCL backend using Intel compilers
RUN cmake -B build \
    -DGGML_SYCL=ON \
    -DCMAKE_C_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release -j$(nproc)

# ------------------------------------------------------------------------------
# Stage 2: Runtime image
# ------------------------------------------------------------------------------
FROM intel/oneapi-basekit:2025.0.0-0-devel-ubuntu22.04

# Install Python and arc-llama dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the built llama-server binary
COPY --from=llama-builder /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
RUN chmod +x /usr/local/bin/llama-server

# Install arc-llama from the local source tree
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e ".[tui]"

# Runtime environment
ENV SYCL_CACHE_PERSISTENT=0
ENV ARC_LLAMA_SERVER=/usr/local/bin/llama-server

# Default port
EXPOSE 11437

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:11437/health || exit 1

# Default entrypoint runs init + serve if no config exists, else just serve
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["serve"]

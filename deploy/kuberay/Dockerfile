# driver and workers share this image, so ray and python cannot drift apart
FROM ghcr.io/prefix-dev/pixi:0.84.1-noble

ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
WORKDIR /app

# the driver's JIT rejects PTX from a newer CUDA; up.sh passes its version
ARG CUDA_MAX=13.2

# hydride has no wheel for this interpreter and builds a C extension
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml MANIFEST.in README.md ./
# stub package, so editing src does not redo the solve
RUN mkdir -p src/openplaceholder \
 && touch src/openplaceholder/__init__.py \
 && pixi add "cuda-version<=${CUDA_MAX}" \
 && rm -rf .pixi/envs/default \
 && pixi install -e dev \
 && (pixi clean cache --yes || true)
COPY src ./src

COPY deploy/kuberay/*.py deploy/kuberay/*.toml /opt/
ENV PATH=/app/.pixi/envs/dev/bin:$PATH

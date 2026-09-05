# Build context: whl_docker_rocm/ (no source tree or model files are copied).
ARG UBUNTU_VERSION=22.04
ARG BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
FROM ${BASE_IMAGE}

ARG ROCM_VERSION=10.0.0
COPY install-sdk.sh /tmp/install-sdk.sh
RUN bash /tmp/install-sdk.sh "${ROCM_VERSION}" && rm /tmp/install-sdk.sh

ENV PATH="/opt/rocm-venv/bin:${PATH}"
WORKDIR /workspace/fastllm
# Mount the source checkout here and pass the usual wheel-builder arguments.
ENTRYPOINT ["bash", "/workspace/fastllm/make_whl_rocm.sh"]

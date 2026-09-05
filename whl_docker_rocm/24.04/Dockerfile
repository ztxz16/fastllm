# Build with: docker build -f whl_docker_rocm/24.04/Dockerfile whl_docker_rocm
# Ubuntu 22.04 is the release baseline; 24.04 wheels need a newer host libc.
ARG UBUNTU_VERSION=24.04
ARG BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
FROM ${BASE_IMAGE}

ARG ROCM_VERSION=10.0.0
COPY install-sdk.sh /tmp/install-sdk.sh
RUN bash /tmp/install-sdk.sh "${ROCM_VERSION}" && rm /tmp/install-sdk.sh

ENV PATH="/opt/rocm-venv/bin:${PATH}"
WORKDIR /workspace/fastllm
ENTRYPOINT ["bash", "/workspace/fastllm/make_whl_rocm.sh"]

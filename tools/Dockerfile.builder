FROM moge-3-panorama-cli:latest

USER root
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENTRYPOINT []
CMD ["/bin/bash"]

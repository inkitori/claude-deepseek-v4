FROM python:3.12-slim

ARG AGENT_UID=2010
ARG AGENT_GID=2010

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev libopenmpi-dev libomp-dev \
        git curl ca-certificates \
        build-essential \
        procps less vim-tiny \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

RUN groupadd -g ${AGENT_GID} agent \
    && useradd -m -u ${AGENT_UID} -g ${AGENT_GID} -s /bin/bash agent \
    && mkdir -p /workspace/work /workspace/logs \
    && chown -R agent:agent /workspace /home/agent

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh

USER agent
ENV HOME=/home/agent

# Claude Code via official installer (drops to /home/agent/.local/bin/claude)
RUN curl -fsSL https://claude.ai/install.sh | bash

ENV PATH=/home/agent/.local/bin:/usr/local/bin:/usr/bin:/bin

WORKDIR /workspace/work

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

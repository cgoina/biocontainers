FROM  docker.io/multifish/biocontainers-dask:2023.10.1

ARG TARGETPLATFORM

RUN dnf group install -y "Development Tools"

COPY conda-env.yml /tmp/

RUN mamba env update -n base -f /tmp/conda-env.yml

ENV CELLPOSE_LOCAL_MODELS_PATH=

WORKDIR /opt/scripts/cellpose

COPY scripts/python /opt/scripts/cellpose

RUN pip install .

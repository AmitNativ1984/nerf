ARG BASE_IMAGE
FROM $BASE_IMAGE as base

ENV SHELL /bin/bash
SHELL [ "/bin/bash", "-c" ]

ENV DEBIAN_FRONTEND=noninteractive

RUN echo ${DEBIAN_FRONTEND}

RUN apt-get update &&\ 
    apt-get install --no-install-recommends -y \
    python3-tk

RUN pip3 install \
    matplotlib \
    pyyaml \
    imageio \
    tqdm \
    wget   



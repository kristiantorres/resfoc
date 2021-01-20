# TODO Have dockerfile inside a git clone directory.
# TODO DASK implementation in a real folder outside container.
FROM python:3.8

RUN apt-get update && \
    apt-get install -y g++ git libfftw3-single3 libfftw3-dev && \
    apt-get -y install make vim wget libtbb-dev && \
    apt-get -y clean

# TODO Consider Miniconda instead...
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh && \
    /bin/bash Anaconda3-2020.02-Linux-x86_64.sh -b -p /opt/anaconda && \
    rm Anaconda3-2020.02-Linux-x86_64.sh

ENV PATH /opt/anaconda/bin:$PATH

COPY ./environment.yml .

# Install ANACONDA Environment
RUN conda env create -f environment.yml && \
    conda clean -afy

ENV PATH /opt/anaconda/envs/smb_env/bin:$PATH
ENV CONDA_PREFIX /opt/anaconda/envs/smb_env

# Download ISPC
RUN wget https://github.com/ispc/ispc/releases/download/v1.13.0/ispc-v1.13.0-linux.tar.gz && \
    tar -xvf ispc-v1.13.0-linux.tar.gz -C /opt

ENV PATH /opt/ispc-v1.13.0-linux/bin:$PATH

## Download git repositories
RUN mkdir /opt/scaas && \ 
          git clone https://github.com/ke0m/scaas.git  /opt/scaas && \
          cd /opt/scaas/scaas/src && \
          make

ENV CPLUS_INCLUDE_PATH $CPLUS_INCLUDE_PATH:/opt/scaas/scaas/src

RUN mkdir /opt/resfoc && \ 
          git clone --recurse-submodules https://github.com/kristiantorres/resfoc.git /opt/resfoc && \
          cd /opt/resfoc/resfoc/src && \
          make && \
          cd /opt/resfoc/velocity/src && \
          make

## Set paths
ENV PYTHONPATH $PYTHONPATH:/opt/scaas
ENV PYTHONPATH $PYTHONPATH:/opt/resfoc

## Start Jupyter notebook
WORKDIR /opt/resfoc/notebooks

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run the Jupyter notebook
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]


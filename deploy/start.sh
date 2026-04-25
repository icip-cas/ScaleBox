#!/usr/bin/env bash

# python 3.11 & poetry
# Install miniconda, if conda is not installed
if ! command -v conda &> /dev/null
then
    echo "conda not found, installing miniconda"
    bash ./scripts/install-miniconda.sh 3.11
    PATH="/root/miniconda3/bin:${PATH}"
    wget https://veml.tos-cn-beijing.volces.com/condarc -O ~/.condarc && \
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    conda init bash
    source ~/.bashrc
fi

# Install poetry if not installed, need to set proxy
# export https_proxy=xxxx
if ! command -v poetry &> /dev/null
then
    echo "poetry not found, installing poetry"
    curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.7.0 python3 -
    export PATH=~/.local/bin:$PATH
fi

mkdir -p environments

# Install python runtime, if `environments/sandbox-runtime` is not found
if [ ! -d "environments/sandbox-runtime" ]
then
    echo "python runtime not found, installing python runtime"
    cd runtime/python
    bash install-python-runtime.sh
    cd -
else
    echo "python runtime found"
    source activate environments/sandbox-runtime
    conda deactivate
fi

# Start sandbox
conda create -n sandbox -y python=3.12
source ~/.bashrc
conda activate sandbox
poetry install
mkdir -p docs/build
make run-online

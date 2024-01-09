FROM ubuntu:22.04

WORKDIR /usr/src/app
COPY . /usr/src/app

ENV WANDB_API_KEY=934436ad14ceb55b75a7917bc289ec0ac28246e2 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:graphics-drivers && \
    apt install nvidia-driver-535 -y && \
    apt install python3.11 python3-pip -y

RUN pip install poetry==1.7.1
RUN poetry install
CMD ["poetry", "run", "python", "wikibot/main.py"]

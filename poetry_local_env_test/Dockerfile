FROM python:3.9-slim-buster

# Use with --build-arg STAGE=dev
ARG STAGE=prod
ENV STAGE=${STAGE}

WORKDIR /app

RUN pip3 install -U pip setuptools wheel poetry
COPY poetry.lock pyproject.toml ./
RUN poetry export --without-hashes -f requirements.txt $(test "${STAGE}" = dev && echo "--dev") -o requirements.txt \
    && pip3 install -q -r requirements.txt

COPY src/ src/
COPY cli.py /app/

CMD python cli.py check

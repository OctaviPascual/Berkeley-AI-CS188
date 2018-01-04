FROM python:2.7-alpine

ENV PYTHONUNBUFFERED=1

RUN mkdir -p /opt/Berkeley-AI-CS188
WORKDIR /opt/Berkeley-AI-CS188

COPY main.py .
COPY project0-tutorial project0-tutorial
COPY project1-search project1-search

ENTRYPOINT [ "python", "main.py" ]

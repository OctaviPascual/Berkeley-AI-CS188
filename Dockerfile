FROM python:2.7-alpine

ENV PYTHONUNBUFFERED=1

RUN mkdir -p /opt/Berkeley-AI-CS188
WORKDIR /opt/Berkeley-AI-CS188

COPY main.py .
COPY project0-tutorial project0-tutorial
COPY project1-search project1-search
COPY project2-multiagent project2-multiagent
COPY project3-reinforcement project3-reinforcement

ENTRYPOINT [ "python", "main.py" ]

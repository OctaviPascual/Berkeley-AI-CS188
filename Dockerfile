FROM frolvlad/alpine-miniconda2

ENV PYTHONUNBUFFERED=1

RUN apk add --no-cache python2-tkinter
RUN conda install -y tensorflow

RUN mkdir -p /opt/Berkeley-AI-CS188
WORKDIR /opt/Berkeley-AI-CS188

COPY main.py main.py
COPY project0-tutorial project0-tutorial
COPY project1-search project1-search
COPY project2-multiagent project2-multiagent
COPY project3-reinforcement project3-reinforcement
COPY project4-bayesNets project4-bayesNets
COPY project5-tracking project5-tracking
COPY project6-classification project6-classification

ENTRYPOINT [ "python", "main.py" ]

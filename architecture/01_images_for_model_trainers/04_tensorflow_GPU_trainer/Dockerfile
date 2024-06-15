
#FROM continuumio/miniconda3:23.3.1-0
FROM tensorflow/tensorflow:nightly-gpu
WORKDIR /home/app

RUN apt-get update
RUN apt-get install unzip
RUN apt-get install nano
RUN apt install curl -y

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

COPY requirements.txt /dependencies/requirements.txt
RUN pip install --ignore-installed -r /dependencies/requirements.txt

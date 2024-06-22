FROM continuumio/miniconda3

WORKDIR /home/app

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

COPY requirements.txt /dependencies/requirements.txt
RUN pip install -r /dependencies/requirements.txt

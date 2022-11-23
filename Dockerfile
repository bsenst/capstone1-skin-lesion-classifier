# syntax=docker/dockerfile:1

FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

# TODO CMD [ "python", "flask_app.py", "run", "--host=127.0.0.1:9696"]
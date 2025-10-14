FROM python:3.11-slim

COPY . /app/

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install -r requirements.txt

# Open two port for api and chatbox app
EXPOSE 30000 30001
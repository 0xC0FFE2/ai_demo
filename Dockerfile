FROM python:3.10-slim

WORKDIR /app

RUN apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python3 download_model.py

COPY app.py .

EXPOSE 8000
CMD ["python3", "app.py"]
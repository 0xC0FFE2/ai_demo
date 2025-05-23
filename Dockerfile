FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 모델 다운로드 스크립트 실행
COPY download_model.py .
RUN python3 download_model.py

# 애플리케이션 파일 복사
COPY app.py .

# 포트 노출 및 실행 명령
EXPOSE 8000
CMD ["python3", "app.py"]
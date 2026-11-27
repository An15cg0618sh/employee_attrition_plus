FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    DATA_PATH=/app/data/employee_data.csv \
    ARTIFACT_DIR=/app/artifacts \
    MONITORING_DIR=/app/artifacts/monitoring \
    ALLOW_ARTIFACT_WRITES=true

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py train.py ./
COPY src ./src
COPY data ./data
COPY .streamlit ./.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

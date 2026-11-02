FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "paho-mqtt>=2,<3" "influxdb-client>=1.40" "pyyaml>=6" "psycopg[binary]>=3.1"
COPY collector.py coverage.py report.py backfill.py ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "collector.py"]

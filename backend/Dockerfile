# Foam-workflow API (Flask + OpenCV + HQ-SAM on CPU). Build context = backend/.
# The checkpoint is NOT in the image: mount it (see compose.prod.yml) or set HQSAM_CKPT.
FROM python:3.12-slim
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000", "--preload"]

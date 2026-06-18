FROM python:3.12-slim

WORKDIR /app

# System deps: PyMuPDF (libmupdf-dev), the handwriting font builder
# (potrace traces the ink), and OpenCV's runtime (libglib2.0-0).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    potrace \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads outputs

ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]

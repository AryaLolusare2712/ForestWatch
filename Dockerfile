FROM python:3.11-slim
WORKDIR /app

# Rasterio needs Expat; OpenCV/YOLO needs libGL and GLib when drawing or
# preprocessing images in the minimal Render container.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "forestwatch.api:app", "--host", "0.0.0.0", "--port", "8000"]

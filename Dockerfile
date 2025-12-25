FROM python:3.11-slim

# ===== system deps for rasterio / gdal =====
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libexpat1 \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# ===== env =====
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# ===== install python deps =====
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ===== copy app =====
COPY . .

# ===== start =====
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]

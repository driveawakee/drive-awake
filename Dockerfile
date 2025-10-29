FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN dnf install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port (adjust if your app uses different port)
EXPOSE 8000

# Run the application
CMD ["python", "Final.py"]

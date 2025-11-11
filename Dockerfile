# Start from a robust, stable Debian base image
FROM debian:bookworm-slim

# Set non-interactive mode for installation
ENV DEBIAN_FRONTEND=noninteractive

# --- CRITICAL SYSTEM PACKAGE INSTALLATION ---
# ADDING build-essential to ensure compilation of Python packages works
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# 2. Install Python dependencies (pip)
# ERROR is here: process "/bin/sh -c pip3 install --no-cache-dir -r requirements.txt" did not complete successfully: exit code: 1
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# 4. Define the command to run the Streamlit app
CMD ["python3", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

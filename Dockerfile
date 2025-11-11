# Start from a robust, stable Debian base image
FROM debian:bookworm-slim

# Set non-interactive mode for installation
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install necessary system dependencies (apt-get)
# This includes the Python runtime and required development tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
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
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 3. Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# 4. Define the command to run the Streamlit app
# We use python3 instead of the generic python command here for explicit control
CMD ["python3", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

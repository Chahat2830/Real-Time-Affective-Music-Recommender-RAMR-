# Use a slightly less-minimal Debian base image for better compatibility
# with complex system libraries like those required by deepface/OpenCV.
FROM python:3.10

# Set environment variable to prevent interactive apt-get prompts
ENV DEBIAN_FRONTEND=noninteractive

# --- CRITICAL SYSTEM PACKAGE INSTALLATION ---
# Install system dependencies in one command for robustness and cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Define the command to run your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

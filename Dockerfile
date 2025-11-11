# Start from a stable Python base image
FROM python:3.10-slim

# Set environment variable to prevent interactive apt-get prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (must be done in one RUN command)
# This installs the packages and cleans up the cache immediately
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the default Streamlit port (Render will handle the mapping)
EXPOSE 8501

# Define the command to run your Streamlit app
# This runs Streamlit and ensures it listens on all interfaces (0.0.0.0)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]# Start from a stable Python base image
FROM python:3.10-slim

# Set environment variable to prevent interactive apt-get prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (must be done in one RUN command)
# This installs the packages and cleans up the cache immediately
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the default Streamlit port (Render will handle the mapping)
EXPOSE 8501

# Define the command to run your Streamlit app
# This runs Streamlit and ensures it listens on all interfaces (0.0.0.0)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

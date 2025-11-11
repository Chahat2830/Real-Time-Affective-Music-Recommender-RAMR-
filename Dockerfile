# Start from a stable Python base image (using 3.10 as an example)
FROM python:3.10-slim

# Install system dependencies needed by OpenCV/deepface
# The backslashes allow the command to span multiple lines for readability
# We install all necessary packages in one RUN command to streamline the build
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    rm -rf /var/lib/apt/lists/*

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
# This is the equivalent of your old Start Command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

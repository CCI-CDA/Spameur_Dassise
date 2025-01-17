FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install build dependencies for bcrypt
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make port 5600 available to the world outside this container
EXPOSE 5600

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5600"]
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    apt-get clean

# Set Python path
ENV PYTHONPATH=/app/backend

# Set working directory
WORKDIR /app

# Copy ALL files
COPY . .

# Install requirements
RUN pip3 install --no-cache-dir -r backend/requirements.txt

# Expose port
EXPOSE 8000

# Correct launch command
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

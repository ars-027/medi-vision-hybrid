# 1. Base Image: Use an official Python runtime with PyTorch and CUDA support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# 2. Set the working directory in the container
WORKDIR /app

# 3. Prevent Python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 4. Install system dependencies required for OpenCV/Matplotlib
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your project files into the container
COPY medivision_hybrid_clinical_best.pth .
COPY app.py .

# 7. Expose the port the app runs on
EXPOSE 8000

# 8. Command to run the engine
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

# Create a non-root user with UID 1000 (Hugging Face default)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set up working directory in user's home
WORKDIR $HOME/app

# Copy requirements file first to cache layers
COPY --chown=user requirements.txt .

# Install dependencies (uses CPU-only PyTorch to fit in memory)
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# Copy all project files into the container
COPY --chown=user . .

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Start FastAPI server
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]

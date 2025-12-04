# Use Python 3.10
FROM python:3.10

# Set working directory
WORKDIR /app

# Install Git LFS (Needed to pull your large vector file)
RUN apt-get update && apt-get install -y git-lfs && git lfs install

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . .

# Build the Database locally inside the container
# (This creates the 'local_chroma_db' folder on the server)
RUN python setup_local_db.py

# Create a non-root user (Security requirement for Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose the port Hugging Face expects
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
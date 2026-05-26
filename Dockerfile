# Use a slim Python 3.10 base image to keep the image small
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only requirements first (layer caching — speeds up rebuilds)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app on container start
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

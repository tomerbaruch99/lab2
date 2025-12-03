FROM python:3.11-slim

# Set UTF-8 locale
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Generate UTF-8 locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY City_Scraper.py .
COPY api_server.py .

# Expose port
EXPOSE 8081

# Run the application
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8081"]


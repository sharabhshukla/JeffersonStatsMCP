FROM python:3.11-slim
LABEL authors="sharabhshukla"

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install uv
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
ENTRYPOINT ["python", "mcpserver.py"]

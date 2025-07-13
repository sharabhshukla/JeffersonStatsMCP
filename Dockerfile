FROM python:3.11-slim
LABEL org.opencontainers.image.source="https://github.com/sharabhshukla/jeffersonstatsmcp"
LABEL org.opencontainers.image.url="https://github.com/sharabhshukla/jeffersonstatsmcp"
LABEL org.opencontainers.image.documentation="https://github.com/sharabhshukla/jeffersonstatsmcp"
LABEL org.opencontainers.image.title="JeffersonStats MCP"
LABEL org.opencontainers.image.description="A comprehensive MCP server providing 40+ statistical analysis tools for AI assistants"
LABEL org.opencontainers.image.version="0.1.0"

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

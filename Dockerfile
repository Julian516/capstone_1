FROM python:3.11-slim

WORKDIR /app

# Install system deps if needed (kept minimal here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model artifacts
COPY src ./src
COPY models ./models

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]

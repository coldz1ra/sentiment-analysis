FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
EXPOSE 8000 8501
CMD ["bash","-lc","uvicorn api.main:app --host 0.0.0.0 --port 8000"]

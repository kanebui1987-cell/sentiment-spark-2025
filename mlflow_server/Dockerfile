FROM python:3.10-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir mlflow boto3 psycopg2-binary

EXPOSE 5000

CMD ["mlflow", "server", 
     "--backend-store-uri", "sqlite:///mlflow.db",
     "--default-artifact-root", "/mlflow_artifacts",
     "--host", "0.0.0.0",
     "--port", "5000"]

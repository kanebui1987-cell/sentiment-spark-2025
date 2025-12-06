# mlops-pipeline-spark

MLOps pipeline using Spark MLlib, MLflow, FastAPI, Docker and GitHub Actions.

## Main components

- `spark/train/train_multiple_models.py`:
  Train multiple models (LR, RF, GBT, LinearSVC), log to MLflow and save the best model to HDFS.
- `spark/inference/batch_predict.py`:
  Batch prediction using the best model from HDFS.
- `spark/streaming/kafka_stream_predict.py`:
  Spark Structured Streaming job that reads from Kafka and outputs predictions.
- `api/sentiment_api.py`:
  FastAPI service loading a trained Spark PipelineModel from `/app/model`.
- `mlflow_server/`:
  Simple MLflow tracking server container.
- `.github/workflows/`:
  CI and CD pipelines (DEV & PROD) using GitHub Actions + SSH deployment.

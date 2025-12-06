from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

class PredictRequest(BaseModel):
    text: str

class HealthResponse(BaseModel):
    status: str

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("SentimentAPI")
    .getOrCreate()
)

try:
    model = PipelineModel.load("/app/model")
    MODEL_LOADED = True
except Exception as exc:  # noqa: F841
    MODEL_LOADED = False
    model = None


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok" if MODEL_LOADED else "model_not_loaded")


@app.post("/predict")
def predict(req: PredictRequest):
    if not MODEL_LOADED:
        return {"error": "Model is not loaded. Please deploy a trained model."}

    df = spark.createDataFrame([(req.text,)], ["text"])
    pred_row = model.transform(df).select("prediction").collect()[0]
    return {"text": req.text, "prediction": int(pred_row["prediction"])}

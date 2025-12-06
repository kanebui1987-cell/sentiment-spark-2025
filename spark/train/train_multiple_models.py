import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    LinearSVC,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from spark.utils.preprocess import basic_text_cleaning
from spark.utils.mlflow_utils import init_mlflow


def main():
    spark = (
        SparkSession.builder
        .appName("TrainMultipleSentimentModels")
        .getOrCreate()
    )

    tracking_uri = "http://name-node-01:5000"
    experiment_name = "Sentiment140_MultiModel"
    init_mlflow(tracking_uri, experiment_name)

    df = (
        spark.read.csv(
            "hdfs:///data/sentiment140.csv",
            header=False,
            inferSchema=True,
        )
        .toDF("label", "id", "date", "query", "user", "text")
    )

    df = basic_text_cleaning(df, text_col="text", output_col="clean")

    tokenizer = Tokenizer(inputCol="clean", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    tf = HashingTF(inputCol="filtered", outputCol="tf")
    idf = IDF(inputCol="tf", outputCol="features")

    candidates = [
        ("LogisticRegression", LogisticRegression(maxIter=30, labelCol="label")),
        ("RandomForest", RandomForestClassifier(numTrees=50, labelCol="label")),
        ("GBT", GBTClassifier(maxIter=20, labelCol="label")),
        ("LinearSVC", LinearSVC(maxIter=20, labelCol="label")),
    ]

    evaluator = MulticlassClassificationEvaluator(
        predictionCol="prediction",
        labelCol="label",
        metricName="accuracy",
    )

    best_model = None
    best_metric = 0.0
    best_name = None

    for model_name, clf in candidates:
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline(stages=[tokenizer, remover, tf, idf, clf])
            model = pipeline.fit(df)
            preds = model.transform(df)

            accuracy = evaluator.evaluate(preds)

            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.spark.log_model(model, f"{model_name}_spark_model")

            print(f"{model_name} accuracy = {accuracy:.4f}")

            if accuracy > best_metric:
                best_metric = accuracy
                best_model = model
                best_name = model_name

    if best_model is None:
        raise RuntimeError("No model successfully trained.")

    print(f"Best model: {best_name} with accuracy={best_metric:.4f}")

    best_model.write().overwrite().save("hdfs:///models/best_sentiment_model")

    spark.stop()


if __name__ == "__main__":
    main()

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


def main():
    spark = SparkSession.builder.appName("BatchPredictSentiment").getOrCreate()

    model = PipelineModel.load("hdfs:///models/best_sentiment_model")

    df = spark.read.text("hdfs:///data/new_tweets.txt").toDF("text")

    preds = model.transform(df)

    preds.select("text", "prediction").write.mode("overwrite").csv(
        "hdfs:///output/sentiment_predictions"
    )

    spark.stop()


if __name__ == "__main__":
    main()

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


def main():
    spark = (
        SparkSession.builder
        .appName("KafkaStreamingSentiment")
        .getOrCreate()
    )

    model = PipelineModel.load("hdfs:///models/best_sentiment_model")

    df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", "name-node-01:9092")
        .option("subscribe", "tweets-stream")
        .load()
    )

    text_df = df.selectExpr("CAST(value AS STRING) as text")

    preds = model.transform(text_df).select("text", "prediction")

    query = (
        preds.writeStream.format("console")
        .outputMode("append")
        .option("truncate", "false")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()

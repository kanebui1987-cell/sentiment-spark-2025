from pyspark.sql import DataFrame
from pyspark.sql.functions import regexp_replace, lower, col


def basic_text_cleaning(df: DataFrame, text_col: str = "text", output_col: str = "clean") -> DataFrame:
    """Apply simple text cleaning to a text column.

    - Lowercase
    - Remove URLs
    - Remove extra spaces
    """
    clean_col = output_col
    cleaned = (
        df.withColumn(clean_col, lower(col(text_col)))
          .withColumn(clean_col, regexp_replace(clean_col, r"http\S+|www\.\S+", ""))
          .withColumn(clean_col, regexp_replace(clean_col, "\s+", " "))
    )
    return cleaned

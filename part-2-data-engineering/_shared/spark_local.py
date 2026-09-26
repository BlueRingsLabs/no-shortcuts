"""A quiet local SparkSession for labs: no web UI, no progress bars, few shuffle partitions."""

from __future__ import annotations

import os
import warnings


def local_spark(cores: int = 4, name: str = "lab", shuffle_partitions: int = 8, extra: dict | None = None):
    os.environ.setdefault("PYSPARK_PYTHON", "python3")
    warnings.filterwarnings("ignore", message="PySpark does not yet fully support pandas")
    from pyspark.sql import SparkSession

    b = (SparkSession.builder.master(f"local[{cores}]").appName(name)
         .config("spark.ui.enabled", "false")
         .config("spark.ui.showConsoleProgress", "false")
         .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
         .config("spark.driver.memory", "2g"))
    for k, v in (extra or {}).items():
        b = b.config(k, v)
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

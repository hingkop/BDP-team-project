from pyspark.sql import SparkSession
from pyspark.sql.functions import avg
from pyspark.sql.functions import desc

if __name__ == "__main__":
    spark = SparkSession.builder.appName("Aggregation").getOrCreate()
    df = spark.read.load("hdfs:///user/maria_dev/project/BDF.csv", format="c    sv", sep=",", inferSchema=True, header=True)
    
    avg_hum = df.groupBy("month").agg(avg("Humidity").alias("avg_hum"))
    avg_hum.show()
    
    avg_prs = df.groupBy("month").agg(avg("Pressure").alias("avg_prs")).orde    rBy(desc("avg_prs"))
    avg_prs.show()
    
    avg_temp= df.groupBy("month").agg(avg("Temperature").alias("avg_temp")).    orderBy(desc("avg_temp"))
    avg_temp.show()

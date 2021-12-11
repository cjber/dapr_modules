from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName('Covid Dataframe').getOrCreate()
print(spark.sparkContext.appName)

# 2a: Load csv to DataFrame with header=True
# either use local or cloud stored data depending on use of dataproc
# or running locally
try:
    # gcloud dataproc jobs submit pyspark a1_spark.py --cluster=spark-cluster
    bucket = 'dataproc-staging-europe-west1-239259997526-zdg428u9'
    csv_path = f'gs://{bucket}/covid19.csv'
    df = spark.read.csv(csv_path, sep=',', header=True)
except Exception as e:
    # $SPARK_HOME/bin/spark-submit a1_spark.py
    print(e, ": Running Locally.")
    csv_path = './data/covid19.csv'
    df = spark.read.csv(csv_path, sep=',', header=True)

# 2b: Show as table, print schema
df.show()
df.printSchema()

# 2c: RDD Filter function
# convert df to rdd
rdd = df.rdd

# before filter
rdd.toDF().count()  # total rows
rdd.toDF().select(  # NULL per column
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
).show()

# Filter any row containing missing value
rdd = rdd.filter(lambda row: None not in row.asDict().values())

# after filter
rdd.toDF().count()  # total rows
rdd.toDF().select(  # NULL per column
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
).show()

# 3: Aggregate + GroupBy highest Total Death per country
rdd_totaldeath = rdd.map(lambda x: (x['location'], int(x['total_deaths'])))\
    .groupByKey()\
    .mapValues(list)\
    .map(lambda x: (x[0], max(x[1])))
rdd_totaldeath = spark.createDataFrame(rdd_totaldeath)\
    .orderBy('_2', ascending=False)
print(rdd_totaldeath.show())

# 4: RDD max and min functions
rdd_totalcases = rdd.map(lambda x: (x['location'], int(x['total_cases'])))\
    .groupByKey()\
    .mapValues(list)\
    .map(lambda x: (x[0], max(x[1])))
print(rdd_totalcases.max(lambda x: x[1]))  # max
print(rdd_totalcases.min(lambda x: x[1]))  # min

# (shown list of at least 20 countries)
rdd_totalcases = spark.createDataFrame(rdd_totalcases)\
    .orderBy('_2', ascending=False)
print(rdd_totalcases.show())

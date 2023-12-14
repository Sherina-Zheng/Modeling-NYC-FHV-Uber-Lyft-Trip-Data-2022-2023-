# Databricks notebook source
# MAGIC %md
# MAGIC ## Import All Libraries and Packages Needed

# COMMAND ----------

# Installling additional modules with %pip
%pip install pandas numpy fsspec s3fs boto3 seaborn
%pip install fastparquet

# COMMAND ----------

import pandas as pd
import boto3
import fastparquet

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.feature import Binarizer
from pyspark.ml import Pipeline
# Import the logistic and linear regression models
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
# Import the evaluation module
from pyspark.ml.evaluation import *
# Import the model tuning module
from pyspark.ml.tuning import *
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect Pyspark with AWS S3

# COMMAND ----------

import os
# To work with Amazon S3 storage, set the following variables using
access_key = 'AKIAY5HJQSIK6HYSSXMB'
secret_key = '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'
# Set the environment variables so boto3 can pick them up later
os.environ['AWS_ACCESS_KEY_ID'] = access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
encoded_secret_key = secret_key.replace("/", "%2F").replace("+", "%2B")
# Set aws_region to where S3 bucket was created
aws_region = "us-east-2"
# Update the Spark options to work with our AWS Credentials
sc._jsc.hadoopConfiguration().set("fs.s3a.access.key", access_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.secret.key", secret_key)
sc._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3." + aws_region + ".amazonaws.com")


file_path = 's3://fhv-data-sz/landing/fhvhv_tripdata_2023-01.parquet'
sdf = spark.read.parquet(file_path, header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleaning

# COMMAND ----------

# Drop duplicates if any
sdf = sdf.drop_duplicates()

# COMMAND ----------

# Dropping any null values on 'on_scene_datetime' column where we will create the fearture
sdf = sdf.dropna(subset=['on_scene_datetime'])

# COMMAND ----------

# Checking all columns
display(sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering

# COMMAND ----------

# Create an indexer for the three string-based columns
indexer = StringIndexer(inputCols=["hvfhs_license_num"], outputCols=["license_index"])
sdf = indexer.fit(sdf).transform(sdf)
# Create an encoder for the three indexes
encoder = OneHotEncoder(inputCols=["license_index", "PULocationID", "DOLocationID"],
 outputCols=["Vendor", "PUVector", "DOVecotor"], dropLast=False)
sdf= encoder.fit(sdf).transform(sdf)

# COMMAND ----------

# Create Binarizers for needed columns
# This binarizer states if there was congestion or not, if yes, then the driver might have a higer possibility to be late, vice versa
congestion_binarizer = Binarizer(threshold=0, inputCol="congestion_surcharge", outputCol="if_congestion")
sdf = congestion_binarizer.transform(sdf)

# This binarizer tests if the driver was on scene on time as planned, 
# by comparing the scheduled pickup_datetime and the actual on_scene_datetime
sdf = sdf.withColumn("if_on_time",
                     when(col("on_scene_datetime") <= col("pickup_datetime"), 1.0)
                     .otherwise(0.0))
                     
# Last binarizer is consolidated from the codes below, it's redundant to set up a string column, then transform it into numerals so that it can be put into a binarizer                    
# sdf = sdf.withColumn("if_on_time", 
#                      when(col("on_scene_datetime") <= col("pickup_datetime"), "On Time")
#                      .otherwise("Delayed"))

# # Since 'if_on_time' is a string column in the DataFrame 'sdf'
# # Convert string column to numerical representation using StringIndexer
# string_indexer = StringIndexer(inputCol="if_on_time", outputCol="if_on_time_numeric")
# indexed_sdf = string_indexer.fit(sdf).transform(sdf)

# # Now perform the binarization on the numeric representation
# on_time_binarizer = Binarizer(threshold=0, inputCol="if_on_time_numeric", outputCol="if_on_time_binary")
# sdf = on_time_binarizer.transform(indexed_sdf)

# COMMAND ----------

# Calculate the time difference between request_datetime and pickup_datetime 
# to see how long actually the customer had to wait for the driver to arrive  
sdf = sdf.withColumn("waiting_time_seconds",
                     (unix_timestamp(col("pickup_datetime")) - unix_timestamp(col("request_datetime"))))
sdf = sdf.withColumn("waiting_time_minutes", col("waiting_time_seconds") / 60)

# COMMAND ----------

# Sett up a lable representing the percentage of the tips
sdf = sdf.withColumn("tipPercent", col("tips") / col("base_passenger_fare") )

# Determine if tips are considered "Good" or "Fair"
# Logistic Regression is good at predicting a binary output so we'll eliminate the third classifier
# sdf = sdf.withColumn("tipQuality",
#                      when(col("tipPercent") > 0.15, "Good")
#                      .when((col("tipPercent") >= 0.10) & (col("tipPercent") <= 0.15), "Fair")
#                      .otherwise("Poor"))
# Instead, use:
sdf = sdf.withColumn("tipQuality",
                     when(col("tipPercent") > 0.15, "Good")
                     .when(col("tipPercent") <= 0.15, "Fair"))

# COMMAND ----------

# Check up the schema
sdf.printSchema()

# COMMAND ----------

# Drop rows with any null values across all columns
sdf = sdf.dropna()

# COMMAND ----------

# Check the data frame
display(sdf) 

# COMMAND ----------

# Create an assembler for the individual feature vectors and the float/double columns
assembler = VectorAssembler(inputCols=['Vendor', 'PUVector', 'DOVecotor', 'tipPercent', 'if_congestion', 'if_on_time', 'waiting_time_minutes'],      
                            outputCol='features')

# Apply the assembler to the DataFrame
sdf = assembler.transform(sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Pipeline 

# COMMAND ----------

# Review the transformed features
sdf.select('if_on_time','waiting_time_minutes','tipQuality','if_congestion','license_index','Vendor', 'PUVector', 'DOVecotor', 'features').show(30, truncate=True)

# COMMAND ----------

# Split the data into 70% training and 30% test sets
trainingData, testingData = sdf.randomSplit([0.7, 0.3], seed=42)

# Create a Linear Regression Estimator
lr = LinearRegression(featuresCol='features', labelCol='tips')

# Create the pipeline   Indexer is stage 0 and Linear Regression (linear_reg)  is stage 3
regression_pipe = Pipeline(stages=[indexer, encoder, assembler, lr])

# Create a grid to hold hyperparameters 
grid = ParamGridBuilder()

# Build the parameter grid
grid = grid.build()

# Create a regression evaluator (to get RMSE, R2, RME, etc.)
evaluator = RegressionEvaluator(labelCol='tips')

# Create the CrossValidator using the hyperparameter grid
cv = CrossValidator(estimator=regression_pipe, 
                    estimatorParamMaps=grid, 
                    evaluator=RegressionEvaluator, 
                    numFolds=3)

# Fit the model to the training data
model = lr.fit(trainingData)

# Print the coefficients and intercepts 
print("Coefficients: ", model.coefficients)
print("Intercept: ", model.intercept)


# # Get the best model from all of the models trained
# bestModel = model.bestModel

# # Use the model 'bestModel' to predict the test set
# test_results = bestModel.transform(testData)

# Test the model on the testData
test_results = model.transform(testingData)

# Evaluate the testing data with RSME
RegressionEvaluator(labelCol="tips").evaluate(test_results)

# Show the test results
test_results.select('if_on_time', 'waiting_time_minutes', 'tipQuality', 'if_congestion', 'license_index', 'Vendor', 'PUVector', 'DOVecotor', 'features').show(truncate=False)

# COMMAND ----------

# # Calculate RMSE and R2
# rmse = evaluator.evaluate(test_results, {evaluator.metricName: 'rmse'})
# r2 = evaluator.evaluate(prediction, {evaluator.metricName: 'r2'})
# print(f"RMSE: {rmse}, R-squared: {r2}")

# COMMAND ----------

display(test_results)

# COMMAND ----------

# If the predicted tips are higher than 15% of the base trip fare?
df = test_results.withColumn('is_larger_than_15_percent', 
                   when(col('tipPercent') > 0.15, True).otherwise(False))

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualization 

# COMMAND ----------

# pip install seaborn wordcloud
# Import additional libraries needed to conduct visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Transform Spark Dataframe into Pandas Dataframe
df = test_results.select('tips', 'tipPercent', 'waiting_time_minutes','prediction').toPandas()

# COMMAND ----------

df

# COMMAND ----------

# Calculate frequencies to compare between vendors, and see which vendor is more populor in the market
print('HV0002: Juno')
print('HV0003: Uber')
print('HV0004: Via')
print('HV0005: Lyft')
counts = df.groupBy('hvfhs_license_num').count().filter("hvfhs_license_num IN ('HV0002', 'HV0003', 'HV0004', 'HV0005')").orderBy('hvfhs_license_num')
counts.show()

# COMMAND ----------

# Plotting histogram using Pandas and Matplotlib
plt.hist(counts, bins=20) 
plt.xlabel('Counts')
plt.ylabel('Frequency')
plt.title('Histogram of Counts')
plt.show()

# COMMAND ----------

# Draw a boxplot to see if the waiting time is a good factor of the tips
# Consider only 500 records
subset_df = df.head(500)

# Scatter plot to visualize the relationship
plt.figure(figsize=(8, 6))
plt.scatter(subset_df['waiting_time_minutes'], subset_df['prediction'], alpha=0.5)
plt.title('Waiting Time vs Tip Prediction')
plt.xlabel('Waiting Time')
plt.ylabel('Tip Prediction')
plt.grid(True)
plt.show()


# COMMAND ----------

# Plot tips against predicted tips (predictions)
import seaborn as sns

# The Spark dataframe test_results holds the original 'tips' as well as the 'prediction'
# Select and convert to a Pandas dataframe
df = test_results.select('tips','prediction').toPandas()

# Set the style for Seaborn plots
sns.set_style("white")
 
# Create a relationship plot between tip and prediction
sns.lmplot(x='tips', y='prediction', data=df)

# Set the style for Seaborn plots
sns.set_style("white")

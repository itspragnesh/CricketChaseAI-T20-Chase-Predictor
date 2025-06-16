from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
from pyspark.ml.feature import VectorAssembler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pickle

# Initialize Spark session
spark = SparkSession.builder.appName("T20_Chase_Prediction_ANN").getOrCreate()

# Load dataset
file_path = "C:/Users/pragn/OneDrive/Desktop/Cricket app/ball_by_ball_it20.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Convert 'Date' to date format
df = df.withColumn("Date", to_date(col("Date")))

# Filter only second innings data
df = df.filter(col("Innings") == 2)

# Remove rows where 'Balls Remaining' == 0
df = df.filter(col("Balls Remaining") != 0)

# Define features and target variable
feature_cols = [
    'Runs From Ball', 'Innings Runs', 'Innings Wickets', 'Balls Remaining',
    'Target Score', 'Total Batter Runs', 'Total Non Striker Runs',
    'Batter Balls Faced', 'Non Striker Balls Faced'
]
target_col = "Chased Successfully"

# Split data into training and testing sets
cutoff_date = "2023-01-01"
train_df = df.filter(col("Date") < cutoff_date).drop("Date")
test_df = df.filter(col("Date") >= cutoff_date).drop("Date")

# Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_df = assembler.transform(train_df).select("features", target_col)
test_df = assembler.transform(test_df).select("features", target_col)

# Convert Spark DataFrame to Pandas
train_pd = train_df.toPandas()
test_pd = test_df.toPandas()

# Extract features and labels
X_train = np.vstack(train_pd["features"].apply(lambda x: x.toArray()))
y_train = train_pd[target_col].values
X_test = np.vstack(test_pd["features"].apply(lambda x: x.toArray()))
y_test = test_pd[target_col].values

# Scale features using scikit-learn StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train ANN Model
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test), verbose=1)

# Save the scaler and ANN model
pickle.dump(scaler, open('scaler.pkl', 'wb'))
ann_model.save('ann_model.h5')

# Stop Spark session
spark.stop()
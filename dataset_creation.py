import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Function to generate timestamp
def generate_timestamp(start, end, num_samples):
    time_diff = (end - start).total_seconds()
    timestamps = [start + timedelta(seconds=random.uniform(0, time_diff)) for _ in range(num_samples)]
    timestamps.sort()
    return timestamps

# Settings
num_samples = 10000
start_time = datetime(2024, 1, 1)
end_time = datetime(2024, 12, 31)
sensor_types = ['temperature', 'vibration', 'pressure']
normal_ranges = {
    'temperature': (20, 80),  
    'vibration': (0.1, 5.0),  
    'pressure': (100, 500)    
}

# Generate Data
data = []
timestamps = generate_timestamp(start_time, end_time, num_samples)
for timestamp in timestamps:
    sample = {'timestamp': timestamp}
    for sensor in sensor_types:
        normal_range = normal_ranges[sensor]
        value = random.uniform(*normal_range)
        sample[sensor] = value
    data.append(sample)

# Convert to DataFrame
df = pd.DataFrame(data)

def introduce_anomalies(df, anomaly_fraction=0.03):
    num_anomalies = int(len(df) * anomaly_fraction)
    anomaly_indices = random.sample(range(len(df)), num_anomalies)
    for idx in anomaly_indices:
        sensor = random.choice(sensor_types)
        # Create a spike or drop in the sensor reading
        df.at[idx, sensor] = df.at[idx, sensor] * random.uniform(1.5, 2.5)
    return df

df_with_anomalies = introduce_anomalies(df)
df_with_anomalies.to_csv('created_iot_data_with_anomalies.csv', index=False)




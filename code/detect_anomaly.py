import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Generate sample sensor data
np.random.seed(42)
normal = np.random.normal(20, 2, 200)
anomalies = np.random.normal(30, 1, 10)
sensor_values = np.concatenate([normal, anomalies])
time = np.arange(len(sensor_values))

# Create dataframe
df = pd.DataFrame({"time": time, "value": sensor_values})

# Train Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = model.fit_predict(df[["value"]])

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(df["time"], df["value"], label="Sensor value")
plt.scatter(
    df[df["anomaly"] == -1]["time"],
    df[df["anomaly"] == -1]["value"],
    label="Anomaly",
    marker='x'
)
plt.legend()
plt.title("Sensor Anomaly Detection using Isolation Forest")
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.show()

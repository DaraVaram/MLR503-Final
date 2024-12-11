# -*- coding: utf-8 -*-
"""GenerateData.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-jqnOMb-n7u42O1AV9vZVdHEG5QmJegl
"""

import pandas as pd
import numpy as np

# Define parameters for synthetic data
np.random.seed(42)
timestamps = pd.date_range(start="2020-01-01", end="2022-12-31", freq="H")
cities = ["City_A", "City_B", "City_C", "City_D", "City_E"]
data = []

# Create synthetic data
for city in cities:
    city_data = {
        "Timestamp": timestamps,
        "City_ID": city,
        "Temperature": np.random.normal(loc=20, scale=10, size=len(timestamps)),  # Avg temp in °C
        "Humidity": np.random.uniform(30, 80, size=len(timestamps)),  # Humidity in %
        "Electricity_Consumption": np.random.normal(loc=50, scale=15, size=len(timestamps)),  # kWh
    }
    city_df = pd.DataFrame(city_data)
    city_df["Electricity_Consumption"] = np.clip(city_df["Electricity_Consumption"], 10, 150)
    data.append(city_df)

# Combine all city data
synthetic_data = pd.concat(data, ignore_index=True)

# Save the smaller dataset
file_path = "/content/drive/MyDrive/MLR503 Final Exam Preparation/Time series based stuff/synthetic_time_series_electricity.csv"
synthetic_data.to_csv(file_path, index=False)
file_path

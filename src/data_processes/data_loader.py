"""
This module contains functions to load and save raw data for the mental health model.
"""

import pandas as pd
import kagglehub
import os
def data_loader():
    # Current file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Output file path
    output_path = os.path.normpath(os.path.join(script_dir, '../../data/raw/raw_mental_health_data.csv'))
    # If the dataset already exists locally, load it from there
    if os.path.exists(output_path):
        return pd.read_csv(output_path)
    
    # Download dataset from Kaggle to local path
    path = kagglehub.dataset_download("atharvasoundankar/mental-health-and-lifestyle-habits-2019-2024")
    # Find the CSV file in the downloaded path
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    # Read CSV file into DataFrame
    df = pd.read_csv(os.path.join(path, csv_file))
    # Save DataFrame to CSV
    df.to_csv(output_path, index=False)
    return df

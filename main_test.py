from src.data_processes.data_loader import data_loader
import pandas as pd
def main():
    # Load dataset and save to CSV as raw data
    data_loader()
    # Read the saved raw data to verify
    raw_df = pd.read_csv('data/raw/raw_mental_health_data.csv')


if __name__ == "__main__":
    main()
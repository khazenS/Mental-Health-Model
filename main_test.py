from src.data_processes.data_analyze import show_categoricals_distrubition, show_numerical_distrubition,check_outliers,show_null_counts
from src.data_processes.data_loader import data_loader
import pandas as pd
def main():
    # Load dataset and save to CSV as raw data
    data_loader()
    # Read the saved raw data to verify
    raw_df = pd.read_csv('data/raw/raw_mental_health_data.csv')

    # Our categorical distribution analysis
    categoricals = ['Country','Gender','Exercise Level','Diet Type','Stress Level','Mental Health Condition']
    for column in categoricals:
        show_categoricals_distrubition(raw_df, column)
    
    #Our numerical distribution analysis
    numerical_features = ['Age','Sleep Hours','Work Hours per Week','Screen Time per Day (Hours)','Social Interaction Score','Happiness Score']
    for column in numerical_features:
        show_numerical_distrubition(raw_df, column)
        check_outliers(raw_df, column)
    
    show_null_counts(raw_df)

if __name__ == "__main__":
    main()
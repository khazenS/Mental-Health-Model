import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Analyze categorical column distribution
def show_categoricals_distrubition(df, column_name):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    # Calculate the distribution ratios
    ratios = df[column_name].value_counts(normalize=True)

    # Check for imbalance for every category of column
    for ratio in ratios.items():
        if(ratio[1] > 0.8):
            print(f"Warning: The column '{column_name}' is highly imbalanced. The category '{ratio[0]}' constitutes {ratio[1]*100:.2f}% of the data.")
            return
    # If no imbalance found
    print(f"Column '{column_name}' distribution is compatible for analysis.")

# Analyze numerical column distribution
def show_numerical_distrubition (df, column_name):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Get skewness and kurtosis for normality check
    skewness = df[column_name].skew()
    kurt = df[column_name].kurtosis()

    # Shappiro Test for normality
    stat, p_value = stats.shapiro(df[column_name])

    if p_value > 0.05:
        print(f"The column '{column_name}' follows a normal distribution. (p-value: {p_value:.4f})")
    elif abs(skewness) < 0.5 and abs(kurt) < 2:
        print(f"The column '{column_name}' is uniform distribution. (skewness: {skewness:.2f} and kurtosis: {kurt:.2f}).")
    else:
        print(f"The column '{column_name}'s datas are skewed somewhere and it is not compatible for us. (p-value: {p_value:.4f}, skewness: {skewness:.2f}, kurtosis: {kurt:.2f})")
# Show null value counts for each column
def show_null_counts(df):
    null_counts = df.isnull().sum()
    for colmn, count in null_counts.items():
        if count > 0:
            print(f"Column '{colmn}' has {count} null values.")

# Analyze outliers in numerical columns using IQR method
def check_outliers(df,column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # IQR calculation
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    # Calculate bounds for detecting outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    outliers_count = len(outliers)
    if not outliers.empty:
        print(f"Outliers detected in column '{column_name}' that : {outliers_count} values. Max value is {outliers[column_name].max()} and Min value is {outliers[column_name].min()}.")
    else:
        print(f"No outliers detected in column '{column_name}'.")

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os 
import math
import seaborn as sns
# Initial data analysis function
def initial_data_analyze(df):
    print("DataFrame Head:")
    print(df.head())
    print(60 * '-')
    print("DataFrame Info:")
    print(df.info())
    print(60 * '-')
    print("DataFrame Description:")
    print(df.describe(include='all'))
    print(60 * '-')
    print("DataFrame Types:")
    print(df.dtypes)

def seperate_variables(df):
    """
    We seperate categorical and numerical columns from the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    columns (list): List of all column names in the DataFrame.
    categorical_cols (list): List of categorical column names.
    numerical_cols (list): List of numerical column names.
    """

    columns = df.columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    return columns, categorical_cols, numerical_cols



def show_categoricals_distribution(df, categorical_cols):
    """
    ( This function for only main_data_analyze.py because of visualization saving path)
    Analyzes and visualizes the distribution of categorical columns in the DataFrame.
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    categorical_cols (list): List of categorical column names to analyze.
    
    Returns:None 
    """
    for column_name in categorical_cols:
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
    
    # --- Visualization ---
    num_plots = len(categorical_cols)
    if num_plots > 0:
        n_cols = 3 
        n_rows = math.ceil(num_plots / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() # Flatten the axes array

        for i, col_name in enumerate(categorical_cols):
            if col_name in df.columns:
                ax = axes[i]
                # Draw the plot
                sns.countplot(x=col_name, hue=col_name, data=df, ax=ax, palette='viridis', order=df[col_name].value_counts().index, legend=False)
                
                ax.set_title(col_name)
                ax.tick_params(axis='x', rotation=45)
                ax.set_xlabel('')
                ax.set_ylabel('Count')
                
                # Write the counts on top of the bars
                for container in ax.containers:
                    ax.bar_label(container)
            
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        # Save the figure to the outputs/figure directory
        output_dir = "outputs/figure/"
        os.makedirs(output_dir, exist_ok=True)
        
        save_path = os.path.join(output_dir, "categorical_distributions_combined.png")
        plt.savefig(save_path)
        plt.close()
        
        print(f"All categorical distribution plots saved in a single file: {save_path}")

# Analyze numerical column distribution
def show_numerical_distribution (df, numerical_cols):
    """( This function for only main_data_analyze.py because of visualization saving path)
    Analyzes and visualizes the distribution of numerical columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    numerical_cols (list): List of numerical column names to analyze.
    Returns: None
    """
    for column_name in numerical_cols:
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

    # Visualization
    num_plots = len(numerical_cols)
    if num_plots > 0:
        n_cols = 3
        n_rows = math.ceil(num_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        axes = axes.flatten()  # Flatten the axes array

        for i, col_name in enumerate(numerical_cols):
            if col_name in df.columns:
                ax = axes[i]
                # Plotting with Seaborn
                sns.histplot(data=df, x=col_name, kde=True, ax=ax, color='skyblue',element='step', stat='density', bins=30)
                
                # Labels and title
                ax.set_title(f'{col_name} Distribution', fontsize=12, fontweight='bold')
                ax.set_xlabel('Value', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        # Save the figure to the outputs/figure directory
        output_dir = "outputs/figure/"
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, "numerical_distributions_combined.png")
        plt.savefig(save_path)
        plt.close()
        
        print(f"All numerical distribution plots saved in a single file: {save_path}")

# Show null value counts for each column
def show_null_counts(df):
    null_counts = df.isnull().sum()
    for colmn, count in null_counts.items():
        if count > 0:
            print(f"Column '{colmn}' has {count} null values.")

# Analyze outliers in numerical columns using IQR method
# ( This function for only main_data_analyze.py because of visualization saving path)
def check_outliers(df,numerical_cols):
    for col in numerical_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
        
        # IQR calculation
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        # Calculate bounds for detecting outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Identify outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_count = len(outliers)
        if not outliers.empty:
            print(f"Outliers detected in column '{col}' that : {outliers_count} values. Max value is {outliers[col].max()} and Min value is {outliers[col].min()}.")
        else:
            print(f"No outliers detected in column '{col}'.")
    
    # --- Dynamic Visualization Saving ---

    # Some settings for subplots
    num_plots = len(numerical_cols)
    cols_per_row = 3
    num_rows = math.ceil(num_plots / cols_per_row)

    fig , axes = plt.subplots(num_rows, cols_per_row, figsize=(5 * cols_per_row, 5 * num_rows))

    axes = axes.flatten()  # Flatten in case of multiple rows
    for i, col in enumerate(numerical_cols):
            # Plotting with Seaborn
            sns.boxplot(y=df[col], ax=axes[i], color='skyblue', width=0.5)
            
            # Labels and title
            axes[i].set_title(f'{col}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Values', fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.6)
    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout 
    plt.tight_layout()

    # Save the figure to the outputs/figure directory
    output_dir = "outputs/figure/"
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "all_outliers_boxplot.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Clear memory

    print(f"\nOutlier boxplots saved to {save_path}")
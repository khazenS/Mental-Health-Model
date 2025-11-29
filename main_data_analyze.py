from src.data_processes.data_loader import data_loader
from src.data_processes.data_analyze import initial_data_analyze,show_null_counts,seperate_variables,check_outliers,show_categoricals_distribution,show_numerical_distribution
from src.data_processes.data_clear import convert_na_to_unknown,datas_to_lowercase,strip_leading_trailing_spaces,ordinal_encode_column
def main():
    # Load the raw data
    raw_df = data_loader()
    processed_df = raw_df.copy()

    # Perform initial data analysis
    initial_data_analyze(raw_df)

    # Show null value counts
    print(60 * '-')
    show_null_counts(raw_df)
    print(60 * '-')

    # Separate categorical and numerical columns
    columns , categorical_cols , numerical_cols = seperate_variables(raw_df)

    # Show and save outliers information for numerical columns
    check_outliers(raw_df, numerical_cols)
    print(60 * '-')
    # We wont delete outliers because there is no extreme values. These values are normal in real life and important for the model.
    
    # Data Cleaning Steps
    processed_df = datas_to_lowercase(processed_df)
    processed_df = strip_leading_trailing_spaces(processed_df)

    # Mental Health Condition has null counts. We decide that it will convert na to 'unknown'.
    # We mention this why we do that in notes.
    processed_df = convert_na_to_unknown(processed_df, 'Mental Health Condition')
    
    # Show categorical columns distribution and save the plots
    print('Categorical columns distribution analysis:')
    show_categoricals_distribution(processed_df, categorical_cols)
    # The output looking very well. The distribution of categorical variables is equally distributed.
    print(60 * '-')

    # Show numerical columns distribution and save the plots
    print('Numerical columns distribution analysis:')
    show_numerical_distribution(processed_df, numerical_cols)
    print(60 * '-')
    # Just sleep hours is normal distributed. Other numerical columns are uniform.

    # ---- Data encoding and transformation steps ----
    
    #We will convert the Stress Level and Exercise Level columns to numerical values (1–3) so that they can be used in both the heatmap and the prediction model.
    #We will add the transformed numerical values as new columns while keeping the original columns.
    processed_df = ordinal_encode_column(processed_df, 'Exercise Level', {'low':1 , 'moderate':2, 'high':3})
    processed_df = ordinal_encode_column(processed_df, 'Stress Level', {'low':1 , 'moderate':2, 'high':3})
    # As additional, we will ordinal encode Gender column
    processed_df = ordinal_encode_column(processed_df, 'Gender', {'male':1 , 'female':0,})
    print(processed_df['Gender'].value_counts())
    
    print(60 * '-')

    print(processed_df['Stress Level'].value_counts())

if __name__ == "__main__":
    main()
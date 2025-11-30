import pandas as pd
def convert_na_to_unknown(df, column_name):
    """
    Converts NaN values in the specified column of the DataFrame to 'unknown'.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    column_name (str): The name of the column to process.

    Returns:
    pd.DataFrame: The DataFrame with NaN values in the specified column replaced with 'unknown'.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    df[column_name] = df[column_name].fillna('unknown')
    return df

def datas_to_lowercase(df):
    """
    Converts all string-type columns in the DataFrame to lowercase.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with all string-type columns converted to lowercase.
    """
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()
    return df

def strip_leading_trailing_spaces(df):
    """
    Removes leading and trailing spaces from all string-type columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: The DataFrame with leading and trailing spaces removed from string-type columns.
    """
    for col in df.select_dtypes(include='object').columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()
    return df

def ordinal_encode_column(df, column_name, mapping_dict):
    """
    Ordinally encodes the specified categorical column in the DataFrame using the provided mapping dictionary.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    column_name (str): The name of the categorical column to be encoded.
    mapping_dict (dict): A dictionary mapping original categorical values to ordinal numerical values.

    Returns:
    pd.DataFrame: The DataFrame with the specified column ordinally encoded.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    df[column_name] = df[column_name].map(mapping_dict)
    return df

def one_hot_encode_column(df, column_name, prefix):
    """
    One-hot encodes the specified categorical column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    column_name (str): The name of the categorical column to be one-hot encoded.
    prefix (str): The prefix to use for the new one-hot encoded columns.

    Returns:
    pd.DataFrame: The DataFrame with the specified column one-hot encoded.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    one_hot = pd.get_dummies(df[column_name], prefix=prefix,dtype='int')
    df = pd.concat([df, one_hot], axis=1)
    return df
def df_numeric_scaler(df, column_name, scaler):
    """
    Scales the specified numerical column in the DataFrame using the provided scaler.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    column_name (str): The name of the numerical column to be scaled.
    scaler: An instance of a scaler from sklearn.preprocessing (e.g., StandardScaler, MinMaxScaler).

    Returns:
    pd.DataFrame: The DataFrame with the specified column scaled.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Reshape the data for scaling
    data_to_scale = df[[column_name]].values

    # Fit and transform the data using the provided scaler
    scaled_data = scaler.fit_transform(data_to_scale)

    # Replace the original column with the scaled data
    df[column_name] = scaled_data

    return df

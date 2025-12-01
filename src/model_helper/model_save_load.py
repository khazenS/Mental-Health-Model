import joblib
import os
def model_save(model, file_path):
    """
    Saves the given model to the specified file path using joblib.

    Parameters:
    model: The model to be saved.
    file_path (str): The path where the model will be saved.

    Returns:
    None
    """

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    
    # Create directory if it doesn't exist
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    

    # Save the model using joblib
    joblib.dump(model, file_path)
    print(f"Model saved to: {file_path}")

def model_load(file_path):
    """
    Loads a model from the specified file path using joblib.

    Parameters:
    file_path (str): The path from where the model will be loaded.

    Returns:
    The loaded model.
    """
    model = joblib.load(file_path)
    print(f"Model loaded from: {file_path}")
    return model

def is_model_file_valid(file_path):
    """
    Checks if the model file at the specified path is valid and can be loaded.

    Parameters:
    file_path (str): The path of the model file to be checked.

    Returns:
    bool: True if the model file is valid, False otherwise.
    """
    try:
        joblib.load(file_path)
        return True
    except Exception:
        return False
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def find_best_parameters(params_grid,model, X_train, y_train, cv=5):
    """
    Performs grid search to find the best hyperparameters for the given model.
    
    Parameters:
    params_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.
    model: The machine learning model for which to find the best hyperparameters.
    X_train: Training feature data.
    y_train: Training target data.
    cv (int): Number of cross-validation folds. Default is 5.
    
    Returns:
    best_model: The model with the best found hyperparameters.
    best_params (dict): The best hyperparameters found during the grid search."""
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=params_grid, cv=cv, n_jobs=-1, verbose=0)
    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Retrieve the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    return best_model, best_params

def determine_tuned_mse_r2(y_test, y_pred):
    """
    Evaluates the model's performance using Mean Squared Error (MSE) and R^2 Score.
    
    Parameters:
    y_test: Actual target values.
    y_pred: Predicted target values.
    
    Returns:
    mse (float): Mean Squared Error of the predictions.
    r2 (float): R^2 Score of the predictions.
    """

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2

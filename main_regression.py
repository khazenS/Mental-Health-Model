import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.model_helper.model_save_load import model_save, model_load, is_model_file_valid
from src.model_helper.model_tuning import determine_tuned_mse_r2, find_best_parameters
from src.model_helper.model_visualization import feature_importance_visualization, real_pred_graph
def main():
    # Load the processed data
    df = pd.read_csv("data/processed/processed_df.csv")
    # Separate features and target variable
    y = df['Happiness Score']
    X = df.drop(columns=['Happiness Score'])
    # Seperate train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print('\n')

    decision_tree = None
    dt_params_grid = {
        "max_depth": [3, 5, 7, 10, 20],
    }
    if is_model_file_valid("outputs/model/regression/decesion_tree_model.pkl"):
        decision_tree = model_load("outputs/model/regression/decesion_tree_model.pkl")
    else:
        # Decsision Tree Regressor Model
        decision_tree = DecisionTreeRegressor(random_state=42,max_depth=5)
        # Train the model
        decision_tree.fit(X_train, y_train)

        # Save the trained model
        model_save(decision_tree, "outputs/model/regression/decesion_tree_model.pkl")
    print("\n")

    # Make predictions
    y_pred = decision_tree.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Finding best model parameters using Grid Search
    dt_best_model , dt_best_params = find_best_parameters(dt_params_grid, DecisionTreeRegressor(random_state=42) , X_train, y_train)
    dt_best_y_pred = dt_best_model.predict(X_test)
    # Evaluate the best model
    dt_tuned_mse , dt_tuned_r2 = determine_tuned_mse_r2(y_test, dt_best_y_pred)

    # Save best model
    model_save(dt_best_model, "outputs/model/regression/decision_tree_tuned_model.pkl")
    # Print evaluation metrics
    print("Decision Tree Regressor Performance:")
    print(f"Base Model Mean Squared Error: {mse:.4f}")
    print(f"Base Model R^2 Score: {r2:.4f}")
    print("\n")
    print("Tuned Model Best Parameters:", dt_best_params)
    print(f"Tuned Model Mean Squared Error: {dt_tuned_mse:.4f}")
    print(f"Tuned Model R^2 Score: {dt_tuned_r2:.4f}")
    print("\n")
    # Visualizations

    # Real vs Predicted plot
    real_pred_graph(y_test, dt_best_y_pred, "Decision Tree Regressor", "outputs/figure/regression/dt_real_vs_pred.png")

    # Feature importance
    feature_importance_visualization(X.columns, dt_best_model.feature_importances_, "outputs/figure/regression/dt_feature_importance.png")
    print(60 * '-')

    # -- Random Forest Regressor --
    rf_params_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10, 20],
    }

    rf_model = None
    if is_model_file_valid("outputs/model/regression/random_forest_model.pkl"):
        rf_model = model_load("outputs/model/regression/random_forest_model.pkl")
    else:
        # Train a Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        rf_model.fit(X_train, y_train)
        # Save the trained Random Forest model
        model_save(rf_model, "outputs/model/regression/random_forest_model.pkl")
    
    print("\n")
    # Make predictions with Random Forest
    y_rf_pred = rf_model.predict(X_test)

    # Evaluate Random Forest model
    mse_rf = mean_squared_error(y_test, y_rf_pred)
    r2_rf = r2_score(y_test, y_rf_pred)

    # Find best parameters for Random Forest using Grid Search
    rf_best_model, rf_best_params = find_best_parameters(rf_params_grid, RandomForestRegressor(random_state=42), X_train, y_train)
    rf_best_y_pred = rf_best_model.predict(X_test)
    # Evaluate the best Random Forest model
    rf_tuned_mse, rf_tuned_r2 = determine_tuned_mse_r2(y_test, rf_best_y_pred)

    # Save best model
    model_save(rf_best_model, "outputs/model/regression/random_forest_tuned_model.pkl")
    print("Random Forest Regressor Performance:")
    print(f"Mean Squared Error: {mse_rf:.4f}")
    print(f"R^2 Score: {r2_rf:.4f}")
    print("\n")
    print("Tuned Model Best Parameters:", rf_best_params)
    print(f"Tuned Model Mean Squared Error: {rf_tuned_mse:.4f}")
    print(f"Tuned Model R^2 Score: {rf_tuned_r2:.4f}")
    print("\n")
    # Visualizations for Random Forest
    # Real vs Predicted plot
    real_pred_graph(y_test, rf_best_y_pred, "Random Forest Regressor", "outputs/figure/regression/rf_real_vs_pred.png")
    # Feature importance
    feature_importance_visualization(X.columns, rf_best_model.feature_importances_, "outputs/figure/regression/rf_feature_importance.png")
    print(60 * '-')


if __name__ == "__main__":
    main()
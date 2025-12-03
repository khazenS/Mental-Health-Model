from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from src.model_helper.model_save_load import model_save, model_load, is_model_file_valid
from src.model_helper.model_tuning import determine_tuned_mse_r2, find_best_parameters
from src.model_helper.model_visualization import feature_importance_visualization, real_pred_graph
from xgboost import XGBRegressor
import os
import seaborn as sns
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

    # -- Linear Regression --
    print("---- Linear Regression Model ----")
    lr_model = None
    if is_model_file_valid("outputs/model/regression/linear_regression_model.pkl"):
        lr_model = model_load("outputs/model/regression/linear_regression_model.pkl")
    else:
        # Linear Regression Model
        lr_model = LinearRegression()
        # Train the model
        lr_model.fit(X_train, y_train)
        # Save the trained model
        model_save(lr_model, "outputs/model/regression/linear_regression_model.pkl")

    # Get predictions
    lr_y_pred = lr_model.predict(X_test)

    # Evaluate the model
    lr_mse = mean_squared_error(y_test, lr_y_pred)
    lr_r2 = r2_score(y_test, lr_y_pred)

    print("Linear Regression Performance")
    print(f"Mean Squared Error: {lr_mse:.4f}")
    print(f"R^2 Score: {lr_r2:.4f}")
    print("\n")

    # Visualizations
    real_pred_graph(y_test, lr_y_pred, "Linear Regression", "outputs/figure/regression/lr_real_vs_pred.png")
    # Feature importance
    feature_importance_visualization(X.columns, abs(lr_model.coef_), "outputs/figure/regression/lr_feature_importance.png")
    print(60 * '-')

    # -- Decision Tree Regressor --
    print("---- Decision Tree Regressor Model ----")
    dt_best_model = None
    dt_best_params = None
    dt_params_grid = {
        "max_depth": [3, 5, 7, 10, 20],
    }

    if is_model_file_valid("outputs/model/regression/decesion_tree_model.pkl"):
        dt_best_model = model_load("outputs/model/regression/decesion_tree_model.pkl")
    else:
        # Finding best model parameters using Grid Search
        dt_best_model , dt_best_params = find_best_parameters(dt_params_grid, DecisionTreeRegressor(random_state=42) , X_train, y_train)
        # Save the trained model
        model_save(dt_best_model, "outputs/model/regression/decesion_tree_model.pkl")
 
    dt_best_y_pred = dt_best_model.predict(X_test)
    # Evaluate the best model
    dt_tuned_mse , dt_tuned_r2 = determine_tuned_mse_r2(y_test, dt_best_y_pred)

    # Print evaluation metrics
    print("Decision Tree Regressor Performance:")
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
    print("---- Random Forest Regressor Model ----")
    rf_params_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10, 20],
    }

    rf_best_params = None
    rf_best_model = None
    if is_model_file_valid("outputs/model/regression/random_forest_model.pkl"):
        rf_best_model = model_load("outputs/model/regression/random_forest_model.pkl")
    else:
        # Find best parameters for Random Forest using Grid Search
        rf_best_model, rf_best_params = find_best_parameters(rf_params_grid, RandomForestRegressor(random_state=42), X_train, y_train)

        # Save the trained Random Forest model
        model_save(rf_best_model, "outputs/model/regression/random_forest_model.pkl")
    
    print("\n")
    # Make predictions with Random Forest
    rf_best_y_pred = rf_best_model.predict(X_test)

    # Evaluate the best Random Forest model
    rf_tuned_mse, rf_tuned_r2 = determine_tuned_mse_r2(y_test, rf_best_y_pred)

    print("Random Forest Regressor Performance:")
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

    # -- XGBoost Regressor --
    print("---- XGBoost Regressor Model ----")

    xgb_best_model = None
    xgb_best_params = None
    xgb_params_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7, 10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }

    if is_model_file_valid("outputs/model/regression/xgboost_model.pkl"):
        xgb_best_model = model_load("outputs/model/regression/xgboost_model.pkl")
    else:
        # Finding best model parameters using Grid Search
        xgb_best_model , xgb_best_params = find_best_parameters(xgb_params_grid, XGBRegressor(objective="reg:squarederror", random_state=42) , X_train, y_train)
        # Save the trained model
        model_save(xgb_best_model, "outputs/model/regression/xgboost_model.pkl")
    print("\n")
    xgb_best_y_pred = xgb_best_model.predict(X_test)
    # Evaluate the best model
    xgb_tuned_mse , xgb_tuned_r2 = determine_tuned_mse_r2(y_test, xgb_best_y_pred)
    # Print evaluation metrics
    print("XGBoost Regressor Performance:")
    print("Tuned Model Best Parameters:", xgb_best_params)
    print(f"Tuned Model Mean Squared Error: {xgb_tuned_mse:.4f}")
    print(f"Tuned Model R^2 Score: {xgb_tuned_r2:.4f}")
    print("\n")

    # Visualizations
    # Real vs Predicted plot
    real_pred_graph(y_test, xgb_best_y_pred, "XGBoost Regressor", "outputs/figure/regression/xgb_real_vs_pred.png")
    # Feature importance
    feature_importance_visualization(X.columns, xgb_best_model.feature_importances_, "outputs/figure/regression/xgb_feature_importance.png")
    print(60 * '-')


    # Model Summary Table
    print("\n" + "="*50)
    print("       Regression Model Performance Summary       ")
    print("="*50)

    # Create a summary DataFrame
    summary_data = {
        'Model': ['Linear Regression', 'Decision Tree (Tuned)', 'Random Forest (Tuned)', 'XGBoost (Tuned)'],
        'MSE (Hata)': [lr_mse, dt_tuned_mse, rf_tuned_mse, xgb_tuned_mse],
        'R2 Score': [lr_r2, dt_tuned_r2, rf_tuned_r2, xgb_tuned_r2]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values(by='R2 Score', ascending=False).reset_index(drop=True)

    # Print the summary table
    print(df_summary.to_string(index=False))

    if not os.path.exists("outputs/reports"):
        os.makedirs("outputs/reports")
    
    save_path = "outputs/reports/regression_model_comparison_summary.png"
    # Visulalization Summary 
    plt.figure(figsize=(10, 6))
    sns.barplot(x='R2 Score', y='Model', data=df_summary, palette='mako', hue='Model', legend=False)
    plt.title('Regression Model R2 Score Comparison')
    plt.xlabel('R2 Score')
    plt.xlim(df_summary['R2 Score'].min() - 0.01, df_summary['R2 Score'].max() + 0.01) # Hassas aralık
    plt.tight_layout()

    plt.savefig(save_path)
    print("="*50)

if __name__ == "__main__":
    main()
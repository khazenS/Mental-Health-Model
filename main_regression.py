import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.model_helper.model_save_load import model_save, model_load, is_model_file_valid
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

    decision_tree = None

    if is_model_file_valid("outputs/model/regression/decesion_tree_model.pkl"):
        decision_tree = model_load("outputs/model/regression/decesion_tree_model.pkl")
    else:
        # Decsision Tree Regressor Model
        decision_tree = DecisionTreeRegressor(random_state=42,max_depth=5)
        # Train the model
        decision_tree.fit(X_train, y_train)

        # Save the trained model
        model_save(decision_tree, "outputs/model/regression/decesion_tree_model.pkl")

    # Make predictions
    y_pred = decision_tree.predict(X_test)


    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    # Print evaluation metrics
    print("Decision Tree Regressor Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Visualizations

    # Real vs Predicted plot
    real_pred_graph(y_test, y_pred, "Decision Tree Regressor", "outputs/figure/regression/dt_real_vs_pred.png")


    # Feature importance
    feature_importance_visualization(X.columns, decision_tree.feature_importances_, "outputs/figure/regression/dt_feature_importance.png")

    # [ANALYSIS] Why does the graph look like steps?
    # The Decision Tree makes "discrete" predictions, not continuous ones.
    # It groups people into buckets (leaves) based on rules like "Sleep < 6".
    # Everyone in the same bucket gets the SAME predicted happiness score.
    # This causes the horizontal lines (steps) seen in the scatter plot.

    # [CRITICAL ISSUE] Model Performance - Underfitting
    # The R2 score is negative, indicating the model failed to capture the pattern.
    # A single Decision Tree is too simple for this complex, non-linear human behavior data.
    # It cannot capture the subtle relationships between lifestyle and happiness.

    # [CONCLUSION] Why not use this model alone?
    # Single Decision Trees are prone to high variance and can easily oversimplify (underfit).
    # They struggle with "noisy" real-world data like mental health surveys.
    # We must upgrade to 'Random Forest Regressor'.
    # Random Forest uses an ensemble of trees to average out errors and improve accuracy.

    # -- Random Forest Regressor --


if __name__ == "__main__":
    main()
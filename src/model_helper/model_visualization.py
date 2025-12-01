import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
def feature_importance_visualization(columns, importances, output_path):
    """
    Visualizes and saves the feature importance as a bar plot.

    Parameters:
    columns (list): List of feature names.
    importances (list): List of feature importance scores.
    output_path (str): Path to save the feature importance plot.

    Returns:
    None
    """
    feature_importance = pd.DataFrame({
        'Feature': columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis',hue='Feature', legend=False)
    plt.title('Feature Importance')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)

    print(f"Feature importance plot saved to: {output_path}")


def real_pred_graph(y_test, y_pred, model_name, output_path):
    """
    Plots and saves a comparison graph of real vs predicted values.

    Parameters:
    y_test (array-like): Actual target values.
    y_pred (array-like): Predicted target values.
    model_name (str): Name of the model for the plot title.
    output_path (str): Path to save the comparison plot.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.8, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} : Real vs Predicted Values')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)

    print(f"{model_name} : Real vs Predicted plot saved to: {output_path}")
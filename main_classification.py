import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from src.model_helper.regression_helper import train_random_forest
from src.model_helper.regression_helper import train_xgboost
from src.model_helper.regression_helper import train_lightGBM_classifier
from src.model_helper.regression_helper import train_logistic_regression_classifier
import matplotlib.pyplot as plt
import seaborn as sns
from src.model_helper.model_visualization import feature_importance_visualization,visualize_model_performance
def main():

    # Load the processed data
    df = pd.read_csv('data/processed/processed_df.csv')
    dropped_columns = ['mhc_depression', 'mhc_anxiety' , 'mhc_unknown','mhc_bipolar','mhc_ptsd']

    # We defined x and y values
    X = df.drop(columns=dropped_columns)
    y = df['mhc_depression']

    #Now let’s split x and y into 80% train, 20% test
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= 0.2 , random_state=42 ,stratify=y)
    
    # Train and evaluate models
    logistic_regression_results =train_logistic_regression_classifier(X_train, y_train,X_test,y_test)
    random_forest_results =train_random_forest(X_train, y_train, X_test, y_test)
    xgboost_results = train_xgboost(X_train, y_train, X_test, y_test)
    #lightgbm_results =train_lightGBM_classifier(X_train, y_train,X_test,y_test)
    
    # Compile results
    result_list =[
        random_forest_results,
        xgboost_results,
       # lightgbm_results,
        logistic_regression_results
    ]
    # Create a DataFrame from the results for easier visualization
    df_results = pd.DataFrame(result_list)

    print("\n--- Model Comparison Results ---")
   # Use model performance visualization function
    visualize_model_performance(df_results)

    # Save feature importance visualizations
    for result in result_list:
        model = result['Model']
        importances = None

        # Tree-based models (Random Forest, XGBoost)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        
        # Linear models (Logistic Regression)
        elif hasattr(model, 'coef_'):
            # Negatif katsayılar da önemli olduğu için mutlak değer alırız
            importances = np.abs(model.coef_[0])

        # If importances were successfully obtained, plot them
        if importances is not None:
            feature_importance_visualization(
                X.columns, 
                importances, 
                f"outputs/figure/classification/{result['Model Name'].lower().replace(' ', '_')}_feature_importance.png"
            )


    #The existing columns in your dataset (X) provide almost no information about the target variable you are trying to predict (mhc_depression).
    #This means that the information you have gathered (e.g., age, gender, sleep score) is insufficient to reliably distinguish the presence of depression
    #Your problem is definitely not with model selection or tuning, it lies in your data.

if __name__ == "__main__":
    main()
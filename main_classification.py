import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from src.model_helper.regression_helper_seyma import train_random_forest
from src.model_helper.regression_helper_seyma import train_xgboost
from src.model_helper.regression_helper_seyma import train_lightGBM_classifier
from src.model_helper.regression_helper_seyma import train_logistic_regression_classifier
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv('data/processed/processed_df.csv')
    dropped_columns = ['mhc_depression', 'mhc_anxiety' , 'mhc_unknown','mhc_bipolar','mhc_ptsd']

    X = df.drop(columns='mhc_depression')
    y = df['mhc_depression']

    #print(df['mhc_depression'].value_counts(normalize=True))
    #In the model: 0 = No Depression (Negative) 80.66%, Depression Present (Positive) 19.33%
    #Since the number of depressed patients is very low, the XGBoost model tends to predict “No Depression” more often. Therefore, we added
    #scale_pos_weight to its parameters to increase the weight.

    # Previous CV score (roc_auc): 0.5113 → almost random prediction, slightly able to distinguish the positive class.
    # Current CV score (after adding scale_pos_weight): 0.5049 → almost random prediction, the model’s ability to detect the positive class slightly improved or stayed almost the same.
    # So the difference is very small; scale_pos_weight alone was not sufficient to compensate for class imbalance. 

    #Let’s also run LightGBM, a strong Gradient Boosting model, to confirm that this low score is not unique to XGBoost.

    #(1)We defined x and y values.
    #y is our target column, representing depressed patients
    #x contains all columns except the depression column
    #Now let’s split x and y into 80% train, 20% test

    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= 0.2 , random_state=42 ,stratify=y)
    # Let' show the results.
    random_forest_results =train_random_forest(X_train, y_train, X_test, y_test)
    xgboost_results = train_xgboost(X_train, y_train, X_test, y_test)
    #lightgbm_results =train_lightGBM_classifier(X_train, y_train,X_test,y_test)
    logistic_regression_results =train_logistic_regression_classifier(X_train, y_train,X_test,y_test)
    

    result_list =[
        random_forest_results,
        xgboost_results,
       # lightgbm_results,
        logistic_regression_results
    ]

    df_results = pd.DataFrame(result_list)
    print("\n--- Model Comparison Results ---")
    def visualize_model_performance(df_results):

        # Accuracy, ROC_AUC ve F1_Depression için bar plot
        metrics = ['Accuracy', 'ROC_AUC', 'F1_Depression']

    
        for metric in metrics:
            plt.figure(figsize=(8,5))
            sns.barplot(x='Model', y=metric, data=df_results)
            plt.title(f'Model Comparison - {metric}')
            plt.ylim(0,1)
            plt.ylabel(metric)
            plt.xlabel('Model')
            plt.show()

# Feature importance görselleştirme
    def visualize_feature_importance(results_list, top_n=10):
        for result in results_list:
            if 'Feature_Importances' in result:
                fi_df = pd.DataFrame(result['Feature_Importances'])
                fi_df = fi_df.sort_values('importance', ascending=False).head(top_n)
            
                plt.figure(figsize=(8,6))
                sns.barplot(x='importance', y='feature', data=fi_df, palette='viridis')
                plt.title(f"{result['Model']} - Top {top_n} Feature Importances")
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.show()

   # Kullanımı
    visualize_model_performance(df_results)
    visualize_feature_importance(result_list)



    #The existing columns in your dataset (X) provide almost no information about the target variable you are trying to predict (mhc_depression).
    #This means that the information you have gathered (e.g., age, gender, sleep score) is insufficient to reliably distinguish the presence of depression
    #Your problem is definitely not with model selection or tuning, it lies in your data.

if __name__ == "__main__":
    main()
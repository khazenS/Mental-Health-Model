from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score,confusion_matrix,f1_score
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV

from sklearn.linear_model import LogisticRegression

from src.model_helper.model_tuning import find_best_parameters
from src.model_helper.model_save_load import model_load, is_model_file_valid, model_save

def train_logistic_regression_classifier(X_train, y_train, X_test, y_test):

    print("\n--- Training the Logistic Regression Model ---")
    params_grid = {
        'max_iter': [500, 1000, 1500],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced']
    }

    best_model = None
    best_params = None

    if is_model_file_valid("outputs/model/classification/logistic_regression_model.pkl"):
        # Load the saved model
        best_model = model_load("outputs/model/classification/logistic_regression_model.pkl")
        # Get the model parameters
        best_params = best_model.get_params()
    else:
        #find the best parameters using grid search
        best_model , best_params = find_best_parameters(
            params_grid,
            LogisticRegression(random_state=42),
            X_train,
            y_train
        )
        #Save the best model
        model_save(best_model, "outputs/model/classification/logistic_regression_model.pkl")

    # Make predictions
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    # Calculate evaluation metrics
    acc_score = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1_score_depr = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\n--- Logistic Regression Test Performance ---")
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {acc_score:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"F1 (Depression): {f1_score_depr:.4f}")
    print(f"Confusion Matrix: \n{conf_matrix}")
    print(60 * '-')
    return {
        'Model': best_model,
        'Model Name': 'Logistic Regression',
        'Accuracy': acc_score,
        'ROC_AUC': roc_auc,
        'F1_Depression': f1_score_depr,
        'Confusion_Matrix': conf_matrix.tolist()
    }

def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n--- Training the Random Forest Model ---")

    params_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }

    best_model = None
    best_params = None

    if is_model_file_valid("outputs/model/classification/random_forest_model.pkl"):
        # Load the saved model
        best_model = model_load("outputs/model/classification/random_forest_model.pkl")
        # Get the model parameters
        best_params = best_model.get_params()
    else:
        # Find the best parameters using grid search
        best_model, best_params = find_best_parameters(
            params_grid,
            RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
            X_train,
            y_train
        )
        # Save the best model
        model_save(best_model, "outputs/model/classification/random_forest_model.pkl")
    #Let’s make predictions, and then get the accuracy values
    y_pred = best_model.predict(X_test)

    #Let’s make predictions, and then obtain the accuracy values.
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    acc_score_ = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1_score_depr = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\n--- Random Forest Test Performance ---")
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {acc_score_:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"F1 (Depression): {f1_score_depr:.4f}")
    print(f"Confusion Matrix: \n{conf_matrix}")
    print(60 * '-')

    return {
        'Model': best_model,
        'Model Name': 'Random Forest',
        'Accuracy': acc_score_,
        'ROC_AUC': roc_auc,
        'F1_Depression': f1_score_depr,
        'Confusion_Matrix': conf_matrix.tolist(),
    }


def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n--- Training the XGBoost Model ---")
    #If hyperparameter tuning is not done, the XGBoost algorithm can lead to overfitting because it is very powerful.
    #Why tuning is done first: GridSearchCV takes the base model and trains it multiple times with different settings from the parameter grid.
    #In other words, it does not start training immediately after building the base model; it first establishes the base and then starts the process of finding the best settings.

    # This way, it ensures that the model is optimized for the specific dataset and task at hand, leading to better performance and generalization.
    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        # Performans/Hız Kontrolü
        'n_estimators': [100, 200],      
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.7, 0.8, 1.0],    #Try sampling at 70%, 80%, and 100%
        'colsample_bytree': [0.7, 0.8, 1.0] #Try sampling at 70%, 80%, and 100%
    }

    best_model = None
    best_params = None

    if is_model_file_valid("outputs/model/classification/xgboost_model.pkl"):
        # Load the saved model
        best_model = model_load("outputs/model/classification/xgboost_model.pkl")
        # Get the model parameters
        best_params = best_model.get_params()
    else:
        # Build the XGBoost model with initial parameters
        model = XGBClassifier(objective='binary:logistic',
                            eval_metric = 'logloss',
                            random_state = 42,
                            scale_pos_weight = n_neg / n_pos if n_pos > 0 and n_neg > 0 else 1)

        # Perform grid search to find the best hyperparameters
        cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
        grid_search = GridSearchCV(estimator = model,
                                param_grid = param_grid,
                                scoring='roc_auc',
                                cv=cv,
                                n_jobs=-1,
                                verbose=0)
        grid_search.fit(X_train , y_train)
        #We choose the best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Save the best model
        model_save(best_model, "outputs/model/classification/xgboost_model.pkl")

    # Lets's calculate prediction and probability values.
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]# Get the predicted probabilities for the positive class

    # Calculate evaluation metrics
    acc_score = accuracy_score(y_test,y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1_dep = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\n--- XGBoost Test Performance ---")
    print(f"Best Parameters: {best_params}")
    print(f"Accuracy: {acc_score:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"F1 (Depression): {f1_dep:.4f}")
    print(f"Confusion Matrix: \n{conf_matrix}")
    print(60 * '-')
    
    return {
        'Model': best_model,
        'Model Name': 'XGBoost',
        'Accuracy': acc_score,
        'ROC_AUC': roc_auc,
        'F1_Depression': f1_dep,
        'Confusion_Matrix': conf_matrix.tolist(),
    }
   

def train_lightGBM_classifier(X_train, y_train, X_test, y_test):
    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)
    base_model = LGBMClassifier(objective='binary',
                                min_child_samples=5,
                                min_data_in_leaf=5,
                                random_state=42,
                                scale_pos_weight = n_neg / n_pos if n_pos > 0 and n_neg > 0 else 1,
                                class_weight='balanced',
                                metric='binary_logloss',
                                n_jobs=-1,
                                max_depth=12,
                                num_leaves=64
                                )
    
    param_grid = {
        # Performans/Hız Kontrolü
        'n_estimators': [100, 200],      
        'learning_rate': [0.05, 0.1,0.2],  
        'max_depth': [3, 5,7],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0] 
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc', 
        cv=cv, 
        verbose=1, 
        n_jobs=-1
    )
   
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    print(f"En İyi Parametreler: {best_params}")

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) 
    
    report = classification_report(y_test, y_pred, output_dict=True)


    print(f"\nModel Perform (Test Set - The Best Model):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    conf_matrix = confusion_matrix(y_test, y_pred)

    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print(feature_importances.head(10))
    
    
    return {
        'Model': best_model,
        'Model Name': 'LightGBM',
        'Accuracy': accuracy,
        'ROC_AUC': roc_auc,
        'F1_Depression': f1,
        'Confusion_Matrix': conf_matrix.tolist(),
        }


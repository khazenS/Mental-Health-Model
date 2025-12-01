from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score,confusion_matrix,f1_score
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV

from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier( n_estimators=300,random_state=42, n_jobs=-1 ,class_weight='balanced')
    model.fit(X_train, y_train)

    #Let’s make predictions, and then get the accuracy values
    y_pred = model.predict(X_test)
    #print(y_pred)
    #Let’s make predictions, and then obtain the accuracy values.
    y_proba = model.predict_proba(X_test)[:, 1]
    #print(y_proba) # The model will tell us the probability of a person having depression as a percentage

    accuracy_score_1 = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy_score_1}")
    roc_auc = roc_auc_score(y_test, y_proba)

    report = classification_report(y_test,y_pred,output_dict=True)
    f1_score_depr = report['1']['f1-score']
    print(f"Random Forest Classification Report: \n {report}")

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Random Forest Confusion Matrix: \n {conf_matrix}")

    #We can look at feature importance to see which columns are more important for the model.
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("\n--- Feature Importance Ranking (Top 10)---")
    print(feature_importances.head(10))

    return {
        'Model': 'RandomForest',
        'Accuracy': accuracy_score_1,
        'ROC_AUC': roc_auc,
        'F1_Depression': f1_score_depr,
        'Confusion_Matrix': conf_matrix.tolist(),
        'Feature_Importances': feature_importances.head(10).to_dict('records')
    }


def train_xgboost(X_train, y_train, X_test, y_test):

    n_pos = sum(y_train == 1)
    n_neg = sum(y_train == 0)

    model = XGBClassifier(objective='binary:logistic',
                          eval_metric = 'logloss',
                          random_state = 42,
                          scale_pos_weight = n_neg / n_pos if n_pos > 0 and n_neg > 0 else 1)
    #If hyperparameter tuning is not done, the XGBoost algorithm can lead to overfitting because it is very powerful.
    #Why tuning is done first: GridSearchCV takes the base model and trains it multiple times with different settings from the parameter grid.
    #In other words, it does not start training immediately after building the base model; it first establishes the base and then starts the process of finding the best settings.
    param_grid = {
        # Performans/Hız Kontrolü
        'n_estimators': [100, 200],      
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.7, 0.8, 1.0],    #Try sampling at 70%, 80%, and 100%
        'colsample_bytree': [0.7, 0.8, 1.0] #Try sampling at 70%, 80%, and 100%
    }

    cv = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    grid_search = GridSearchCV(estimator = model,
                               param_grid = param_grid,
                               scoring='roc_auc',
                               cv=cv,
                               n_jobs=-1,
                               verbose=1)
    grid_search.fit(X_train , y_train)
    #We choose the best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best Params:", grid_search.best_params_)
    print("Best CV Score (roc_auc):", grid_search.best_score_)
    # Lets's calculate prediction and probability values.
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]# yüzde kaç olasılıkla depresyon oldugunu soylıcek
    acc_score = accuracy_score(y_test,y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_dep = report['1']['f1-score']
    conf_matrix = confusion_matrix(y_test, y_pred)
    

    # 5. Feature Importance
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print(feature_importances.head(10))

    return {
        'Model': 'XGBoost',
        'Accuracy': acc_score,
        'ROC_AUC': roc_auc,
        'F1_Depression': f1_dep,
        'Confusion_Matrix': conf_matrix.tolist(),
        'Feature_Importances': feature_importances.head(10).to_dict('records')
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
        'Model': 'LightGBM',
        'Accuracy': accuracy,
        'ROC_AUC': roc_auc,
        'F1_Depression': f1,
        'Confusion_Matrix': conf_matrix.tolist(),
        'Best_Params': best_params 
        }



def train_logistic_regression_classifier(X_train, y_train, X_test, y_test):

    print("\n--- Training the Logistic Regression Model ---")

    model = LogisticRegression(
        random_state=42, 
        class_weight='balanced',
        solver='liblinear',    
        max_iter=1000           
    )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc_score = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_score_depr = report['1']['f1-score']
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"\n--- Logistic Regression Test Performance ---")
    print(f"Accuracy: {acc_score:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"F1 (Depression): {f1_score_depr:.4f}")
    print(f"Classification Report: \n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix: \n{conf_matrix}")

    return {
        'Model': 'LogisticRegression',
        'Accuracy': acc_score,
        'ROC_AUC': roc_auc,
        'F1_Depression': f1_score_depr,
        'Confusion_Matrix': conf_matrix.tolist()
    }
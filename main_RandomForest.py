import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("data/processed/processed_df.csv")


y = df["Happiness Score"]

X = df.drop(columns=["Happiness Score"])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,          
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1                 
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(" RANDOM FOREST RESULTS")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

def predict_new_person(model, X_columns, new_data: dict):
    new_df = pd.DataFrame([new_data])
    
    
      
    
    prediction = model.predict(new_df)[0]
    return prediction

trained_rf_model = model
trained_rf_columns = list(X.columns)

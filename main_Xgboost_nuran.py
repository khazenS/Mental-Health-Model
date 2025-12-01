import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


df = pd.read_csv("data/processed/processed_df.csv")


categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
if categorical_cols:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


y = df["Happiness Score"]
X = df.drop(columns=["Happiness Score"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=15,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42,
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n XGBOOST RESULTS")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


def predict_new_person(model, X_columns, new_data: dict):
    new_df = pd.DataFrame([new_data])

    
    for col in X_columns:
        if col not in new_df.columns:
            new_df[col] = 0

    
    new_df = new_df[X_columns]

    prediction = model.predict(new_df)[0]
    return prediction



trained_xgb_model = model
trained_xgb_columns = list(X.columns)

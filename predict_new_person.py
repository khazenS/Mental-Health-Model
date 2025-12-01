import pandas as pd


from main_Linearregression_nuran import trained_lr_model, trained_lr_scaler, trained_lr_columns
from main_RandomForest import trained_rf_model, trained_rf_columns
from main_Xgboost_nuran import trained_xgb_model, trained_xgb_columns



def predict_lr(new_data):
    df = pd.DataFrame([new_data])

    for col in trained_lr_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[trained_lr_columns]

    df_scaled = trained_lr_scaler.transform(df)
    return trained_lr_model.predict(df_scaled)[0]


def predict_rf(new_data):
    df = pd.DataFrame([new_data])

    for col in trained_rf_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[trained_rf_columns]
    return trained_rf_model.predict(df)[0]


def predict_xgb(new_data):
    df = pd.DataFrame([new_data])

    for col in trained_xgb_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[trained_xgb_columns]
    return trained_xgb_model.predict(df)[0]




new_person = {
    'Age': 18,
    'Sleep Hours': 8,
    'Work Hours per Week': 5,
    'Screen Time per Day (Hours)': 4,
    'Social Interaction Score': 6,

    'diet_balanced': 1,
    'diet_junk food': 0,
    'diet_keto': 0,
    'diet_vegan': 0,
    'diet_vegetarian': 0,

    'mhc_anxiety': 1,
    'mhc_bipolar': 0,
    'mhc_depression': 1,
    'mhc_ptsd': 0,
    'mhc_unknown': 1,

    'gender_female': 1,
    'gender_male': 0,
    'gender_other': 0,

    'country_australia': 0,
    'country_brazil': 1,
    'country_canada': 0,
    'country_germany': 0,
    'country_india': 0,
    'country_japan': 0,
    'country_usa': 0
}



print("\n NEW PERSON PREDICTIONS :")

print("Linear Regression:", predict_lr(new_person))
print("Random Forest    :", predict_rf(new_person))
print("XGBoost          :", predict_xgb(new_person))

import pandas as pd
import joblib
import numpy as np
import os

# Raw Data of a New Person
user_raw_datas = [
    {'Age': 18,
    'Sleep Hours': 8,
    'Work Hours per Week': 22,
    'Screen Time per Day (Hours)': 2,
    'Social Interaction Score': 8, 
    'Happiness Score': 9,

    'Stress Level': 1,
    'Exercise Level': 3,

    'diet_balanced': 1,
    'diet_junk food': 0,
    'diet_keto': 0,
    'diet_vegan': 0,
    'diet_vegetarian': 0,

    'mhc_anxiety': 0,
    'mhc_bipolar': 0,
    'mhc_depression': 1,
    'mhc_ptsd': 0,
    'mhc_unknown': 0,

    'gender_female': 0,
    'gender_male': 0,
    'gender_other': 1,

    'country_australia': 0,
    'country_brazil': 0,
    'country_canada': 0,
    'country_germany': 0,
    'country_india': 0,
    'country_japan': 0,
    'country_usa': 1
    },
    {'Age': 60,
    'Sleep Hours': 4,
    'Work Hours per Week': 55,
    'Screen Time per Day (Hours)': 7,
    'Social Interaction Score': 3, 
    'Happiness Score': 3,

    'Stress Level': 3,
    'Exercise Level': 1,

    'diet_balanced': 0,
    'diet_junk food': 1,
    'diet_keto': 0,
    'diet_vegan': 0,
    'diet_vegetarian': 0,

    'mhc_anxiety': 0,
    'mhc_bipolar': 0,
    'mhc_depression': 0,
    'mhc_ptsd': 0,
    'mhc_unknown': 1,

    'gender_female': 0,
    'gender_male': 0,
    'gender_other': 1,

    'country_australia': 0,
    'country_brazil': 0,
    'country_canada': 0,
    'country_germany': 0,
    'country_india': 0,
    'country_japan': 0,
    'country_usa': 1
    }
]

class MentalHealthPredictor:
    def __init__(self):
        # File paths
        self.model_path_reg = 'outputs/model/regression/random_forest_model.pkl' # Happiness Model
        self.model_path_clf = 'outputs/model/classification/xgboost_model.pkl' # Depression Model
        self.scaler_path = 'outputs/scalers/minmax_scaler.pkl'
        self.columns_path = 'outputs/scalers/processed_df_columns.pkl'
        
        # Load Models and Artifacts
        self.reg_model = self._load_file(self.model_path_reg)
        self.clf_model = self._load_file(self.model_path_clf)
        self.scaler = self._load_file(self.scaler_path)
        self.model_columns = self._load_file(self.columns_path)

    # Load a file utility
    def _load_file(self, path):
        if os.path.exists(path):
            return joblib.load(path)
        else:
            raise FileNotFoundError(f"{path} is not found.")
    
    # Preprocess input data
    def preprocess_input(self, user_data):
        """
        Processing the raw user input data to match the model's expected format. 
        """
        # Convert to DataFrame
        df = pd.DataFrame([user_data])
        # Ensure all model columns are present
        scaling_comuns = ['Age', 'Sleep Hours', 'Work Hours per Week', 'Screen Time per Day (Hours)', 'Social Interaction Score', 'Happiness Score']
        # Scale numerical columns
        df[scaling_comuns] = self.scaler.transform(df[scaling_comuns])
        
        return df

    # Predict Happiness Score
    def happiness_score_predict(self, user_data):
        # Preprocess input data
        processed_data = self.preprocess_input(user_data)
        processed_data = processed_data.drop(columns=['Happiness Score'])
        
        # --- CRITICAL FIX: COLUMN ORDER ---
        # We need to ensure that the input DataFrame has the same columns in the same order as the model was trained on.
        expected_features = self.reg_model.feature_names_in_
        processed_data = processed_data.reindex(columns=expected_features, fill_value=0)
        
        
        happiness_score = self.reg_model.predict(processed_data)
        happiness_original = happiness_score[0] * (10 - 1) + 1  # Reverse Min-Max Scaling, we dont have time to scale with scaler for happiness score
        print("Happiness Score Prediction:", round(happiness_original, 2))
        return round(happiness_original, 2)
    
    # Predict Depression Risk
    def depression_risk_predict(self, user_data):
        processed_data = self.preprocess_input(user_data)
        dropped_columns = ['mhc_depression', 'mhc_anxiety' , 'mhc_unknown','mhc_bipolar','mhc_ptsd']
        processed_data = processed_data.drop(columns=dropped_columns)
        
        # --- FIX: COLUMN CLEANUP ---
        # We need to remove both the mhc_ columns AND the Happiness Score (neither were present during training)
        # The easiest way is to use 'reindex' again.
        # It will only keep what the model expects (feature_names_in_).
        
        expected_features = self.clf_model.feature_names_in_
        processed_data = processed_data.reindex(columns=expected_features, fill_value=0)
        depression_risk = self.clf_model.predict(processed_data)
        print("Depression Risk Prediction:", depression_risk[0])
        return depression_risk[0]

def main():    
    # Create predictor instance
    predictor = MentalHealthPredictor()
    
    for i, user in enumerate(user_raw_datas):
        print(f"\n--- Predictions for User {i+1} ---")
        # Get processed user data
        processed_user_data =predictor.preprocess_input(user) 
        # Make Predictions
        predictor.happiness_score_predict(user)
        predictor.depression_risk_predict(user)
if __name__ == "__main__":
    main()
# Mental Health & Lifestyle Analysis Project

  Mental health is a complex issue influenced by various lifestyle, work-related, and social factors that are often difficult to analyze using traditional methods. This project aims to explore the relationship between daily life variables and mental health outcomes using machine learning techniques. By applying both regression and classification models, the study predicts mental health conditions, estimates happiness levels, and identifies the most influential factors affecting mental well-being. The project highlights the potential of machine learning as a decision support tool in understanding and analyzing mental health trends.

  - Predict mental health conditions based on lifestyle and demographic features
  - Estimate an individual’s happiness score using continuous variables
  - Identify which factors have the strongest influence on mental well-being

## Dataset Description
The dataset used in this project is a synthetic mental health dataset designed to simulate real-world scenarios. It contains both numerical and categorical features related to lifestyle, work habits, and psychological well-being.
This is our dataset: https://www.kaggle.com/datasets/atharvasoundankar/mental-health-and-lifestyle-habits-2019-2024/data

·  Demographic Information
 - Country
 - Age
 - Gender

·  Lifestyle Factors 
 - Exercise Level
 - Diet Type
 - Sleep Hours
 - Screen Time per Day (Hours)
 - Social Interaction Score

·  Work & Stress Indicators
  - Work Hours per Week
  - Stress Level

·  Mental Health Indicators
  - Mental Health Condition (e.g., Anxiety, Depression, PTSD)
  - Happiness Score (target variable for regression tasks)

## Data Preprocessing Procedures

HANDLING MISSING VALUES

  - Missing values in the Mental Health Condition feature were treated as an “Unknown” category.
  - This approach avoids data loss while preserving meaningful information.

ENCODING CATEGORICAL VARIABLES
  Categorical features were transformed using One-Hot Encoding, including:
   - Country
   - Gender
   - Diet Type
   - Mental Health Condition

This step converts categorical variables into numerical format suitable for machine learning algorithms.

FEATURE SCALING

  Numerical features such as:
  - Age
  - Sleep Hours
  - Work Hours per Week
  - Screen Time
  - Social Interaction Score
  - Happiness Score

were scaled using Min-Max Normalization, bringing all values into the [0, 1] range.
This prevents features with larger magnitudes from dominating the learning process.

PROCESSED DATASET

  After preprocessing, the final dataset contains only numerical features and is optimized for both regression and classification tasks.
  
The processed dataset is stored in:

 data/processed/processed_df.csv

## Machine Learning Tasks
This project includes multiple modeling approaches:

  Regression Models
  - Predicting Happiness Score

  Classification Models
  - Predicting Mental Health Condition

Each task is implemented in separate scripts for modularity and clarity.
## Model Details and Methodology

Model Architecture This project implements a dual-pipeline approach to analyze mental health data:

1. Classification Pipeline (Depression Risk):
- Goal: Predict the presence of depression (mhc_depression).
- Models: Logistic Regression, Random Forest, XGBoost.
- Setup: 80/20 Stratified Split to handle class imbalance.

2. Regression Pipeline (Happiness Score):
- Goal: Predict the continuous Happiness Score.
- Models: Linear Regression, Decision Tree, Random Forest, XGBoost.
- Optimization: Utilized GridSearchCV to tune hyperparameters (e.g., n_estimators, max_depth, learning_rate).
  
### Motivation

- Comprehensive Benchmarking: We implemented both linear baselines and complex ensemble methods (Tree-based) to empirically determine the best fit for both categorical and continuous targets.
- Performance Optimization: For regression, Grid Search was used to minimize Mean Squared Error (MSE) and maximize $R^2$ scores.
- Explainability: Feature importance analysis is applied across all models to identify the most significant factors contributing to depression risks and happiness levels.
  
## Folder Structure

```text
Mental_Health_Project/
│
├── data/
│   ├── raw/    # Raw Dataset from Kaggle
│   └── processed/    # Processed dataset
│
├── src/  # Source codesand helpers (Data Loader, Analyze vb.)
│
├── outputs/  # Project Outputs
│   ├── model /  # Trained models as .pkl
│   ├── figure/ # Analyze Figures (Heatmap,Feature Importance etc.)
│   └── scalers/ # Data scale files
│
├── main_data_analyze.py # All data information and preprocess steps
├── main_regression.py        # Regression models and outputs
├── main_classification.py    # Classification nodels and outputs
└── predict_user.py           # Real prediction file
```

## Installation

- Open terminal and join project path. Then install requirements

```bash
  pip install -r requirements.txt
```

- After downloaded requirements, you need to analyze and preprocess the raw data. For make it

```python
  python main_data_analyze.py
```
You see all of data information on terminal and If you want to examine about data, you look into outputs/figure/analyze folder. 
All of the figures were saved. Also our raw data was preprocessed, path is data/processed.

## Usage
The project has totally 2 subtask happiness score (regression) and depression risk (classification) prediction.

### Regression 
Again type to terminal for regression models training and outputs of it.

```python
  python main_regression.py
```
You can see all summary about models on terminal and also if you wonder about feature importance,real vs prediction distribution etc. 
look into outputs/figure/regression folder. Model pkl files was saved into outputs/model/regression.

### Classification
Type to terminal for classification models training and outputs of it.

```python
  python main_classification.py
```
You can see all models of outputs on terminal and also if you wonder about feature importance look into outputs/figure/classification folder and 
the summary figure is in outputs/reports.Model pkl files was saved into outputs/model/classification.

### Prediction
There are 2 type of people to compare outputs of models in real_life_predict.py.
```python
  python real_life_predict.py
```
If you want to change the parameters of people, do not change structure of person data. It should like ;
```text
{
    'Age': 18,
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
}
```

- You can see the happiness score and depression risk outputs on terminal.


# Mental Health & Lifestyle Analysis Project

This project is an machine learning app that analysis person's lifestyle, mental condition etc. then predict their happiness score and depression risk.

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

You can see the happiness score and depression risk outputs on terminal.

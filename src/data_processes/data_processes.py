import pandas as pd

#Firstly,We load the dataset.
df = pd.read_csv("../../data/raw/raw_mental_health_data.csv")

#print(df.head())

#Let’s do a basic inspection

#df.info() #There are a total of 3000 rows. Mental Health Condition: 2405 non-null, others:3000 non-null

#print(df.describe())#All numerical columns are within a reasonable range, with no extreme or erroneous values; in other words, there are no outliers.

#print(df.dtypes) # age,Work Hours per Week:int , Sleep Hours,Screen Time per Day (Hours),Social Interaction Score,Happiness Score:float, others:object

#print(df.isnull().sum()) # Mental Health Condition:595 null others:0 null


#We will convert the Stress Level and Exercise Level columns to numerical values (1–3) so that they can be used in both the heatmap and the prediction model.
#We will add the transformed numerical values as new columns while keeping the original columns.
df["Exercise Level Numeric"] = df["Exercise Level"].map({'Low':1 , 'Moderate':2, 'High':3})
#print(df["Exercise Level Numeric"].head()) #İt works correctly

df["Stress Level Numeric"] = df["Stress Level"].map({'Low':1 , 'Moderate':2, 'High':3})

#df.info() # There are 13 column


#We will convert every value in the rows of object-type columns to lowercase.
for col in df.select_dtypes(include=['object']).columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.lower()


#Leading and trailing unnecessary spaces in all object (string) type columns will be removed.
for col in df.select_dtypes(include='object').columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()


# To use this column, we need to convert it to a numerical type. Therefore, we will apply one-hot encoding.”

# One-hot encoding
diet_dummies = pd.get_dummies(df['Diet Type'], prefix='Diet',dtype='int')
#Merge it with the original DataFrame.
df = pd.concat([df, diet_dummies], axis=1)

# Yeni oluşturulan sütunların ilk 10 satırını göster




#print(df['Mental Health Condition'].value_counts()) #For,Mental Health Condition column: anxiety;628 , ptsd;624 , depression;580 , bipolar;573 , 595 null

#önce na değerlerini(none) unnknown olarak ayrı bır kategorı yaptık 
#şimdi one-hot ile mhc sutununun ıcındekı farklı turdekı hastalıkları ayrı sutunlara donuşturucez
df["Mental Health Condition"] = df["Mental Health Condition"].fillna('unknown')
mhc_dummies = pd.get_dummies(df["Mental Health Condition"], prefix='mhc', dtype='int')
df = pd.concat([df, mhc_dummies], axis=1)
#print(df[['Mental Health Condition', 'mhc_anxiety', 'mhc_depression', 'mhc_unknown','mhc_bipolar','mhc_ptsd']].head(10))

#print(df.info())

#For Imputation we can say:
#Missing values could cause the model to learn incorrectly if filled with the mode, so we separated them as 'Unknown' and split them into separate columns;
#This way, the model can independently learn that these examples do not have any of the listed conditions or chose not to specify.

#Eksik değerleri mode ile doldurmak modelin yanlış öğrenmesine yol açacağı için, bunları 'Unknown' olarak ayırıp ayrı sütunlara böldük; 
#Böylece model, bu örneklerin listedeki hastalıklardan hiçbirine sahip olmadığını veya belirtmek istemediğini bağımsız olarak öğrenebilir.



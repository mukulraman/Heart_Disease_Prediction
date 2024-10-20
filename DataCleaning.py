import pandas as pd
import sqlite3

def data_cleaning(df):
    df.replace([''], pd.NA, inplace=True)
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    df['PhysicalHealth'] = pd.to_numeric(df['PhysicalHealth'], errors='coerce')
    Not_Nan=['BMI','PhysicalHealth']

    for mean_column in Not_Nan:
        mean_value = df[mean_column].mean()
        df[mean_column].fillna(mean_value, inplace=True)

    Not_Null=['Smoking','DiffWalking','Sex','Diabetic','GenHealth','SkinCancer','Asthma']

    for column in Not_Null:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)

    cat_features=['HeartDisease','Smoking','AlcoholDrinking','Stroke','DiffWalking','PhysicalActivity','Asthma','KidneyDisease','SkinCancer']
    df[cat_features] = df[cat_features].apply(lambda x: x.map({'Yes': 1, 'No': 0}))

    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    df = pd.get_dummies(df, columns=['Race', 'AgeCategory', 'Diabetic', 'GenHealth'], dtype=int)

    return df
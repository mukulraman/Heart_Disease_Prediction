import pandas as pd
import joblib

from DataCleaning import data_cleaning

from Model_Building_Classification import(
    train_test_split_and_features,
    fit_and_evaluate_model
)    

# Read the Heart Disease Training Data from output_data.csv file in data folder
# If output_data.csv file in data folder doesn't exist, then first run the DB_to_CSV.py script
df = pd.read_csv('data/output_data.csv')
df=data_cleaning(df)

x_train, x_test, y_train, y_test,features=train_test_split_and_features(df)

print(df.head())
model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)

joblib.dump(model, "models/Model_Classifier_Heart.pkl")
joblib.dump(features, "models/Features_Columns.pkl")
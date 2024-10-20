from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from  xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

def train_test_split_and_features(df):
    y = df['HeartDisease']
    x = df.drop('HeartDisease',axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle = False, random_state = 0)
    print(x.head(5))
    scaler=MinMaxScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    print(x.columns)
    features = list(x.columns)
    return x_train, x_test, y_train, y_test,features

def fit_and_evaluate_model(x_train, x_test, y_train, y_test):
    xgb =  XGBClassifier(random_state=0)
    xgb.fit(x_train, y_train)
    xgb_predict = xgb.predict(x_test)
    xgb_conf_matrix = confusion_matrix(y_test, xgb_predict)
    xgb_acc_score = accuracy_score(y_test, xgb_predict)
    print("confussion matrix")
    print(xgb_conf_matrix)
    print("\n")
    print("Accuracy of XGBoost:",xgb_acc_score*100,'\n')
    print(classification_report(y_test,xgb_predict))
    return xgb
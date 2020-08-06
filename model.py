import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def model_train_testing_accuracy():
    # data Loading
    data_set = pd.read_csv('training.csv', sep=';')
    test_data = pd.read_csv('validation.csv', sep=';')

    df_merged = pd.concat([data_set, test_data], axis=1)

    # data preprocessing 
    # Drop Varible 18 for most NANS
    df_merged.drop(['variable18'], inplace=True, axis=1)
    # dropping ALL duplicte values
    df_merged.drop_duplicates(inplace=True)
    # Drop all Nan rows
    df_merged = df_merged.dropna()
    labels = df_merged.iloc[:, -1].values

    # Drop Feautes for most NANS
    df_merged.drop(['classLabel'], inplace=True, axis=1)
    df_merged = pd.get_dummies(df_merged)

    # Data Split
    features = df_merged.iloc[:, :].values
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, shuffle=True)

    # model
    logreg_clf = LogisticRegression()
    # model fit
    model = logreg_clf.fit(x_train, y_train)
    # prediction
    predict = model.predict(x_test)
    print("testing accuracy", accuracy_score(y_test, predict))


def predict_result(data):
    # data Loading
    data_set = pd.read_csv('training.csv', sep=';')
    test_data = pd.read_csv('validation.csv', sep=';')

    df_merged = pd.concat([data_set, test_data], axis=1)

    # data preprocessing 
    # Drop Varible 18 for most NANS
    df_merged.drop(['variable18'], inplace=True, axis=1)
    # dropping ALL duplicte values
    df_merged.drop_duplicates(inplace=True)
    # Drop all Nan rows
    df_merged = df_merged.dropna()
    labels = df_merged.iloc[:, -1].values

    # Drop Feautes for most NANS
    df_merged.drop(['classLabel'], inplace=True, axis=1)
    df_merged = pd.get_dummies(df_merged)

    # Data Split
    features = df_merged.iloc[:, :].values
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, shuffle=True)

    # model
    logreg_clf = LogisticRegression()
    # model fit
    model = logreg_clf.fit(x_train, y_train)
    # prediction
    predict = model.predict(x_test)
    return predict


model_train_testing_accuracy()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
Assuming I am working on the Compltet Data

Notes about the data while cleaning 
The data is'nt balanced (Class inbalance)
yes.    3424
no.      276
There is about 3000 repeted row
After removing duplicates
no.     276
yes.    214
variable18 is the most NAN containg column so I removed it
Normlize variable17, variable14,variable15
Remove the Rest Nan Rows


'''
def model_train_testing_accuracy():
    # data Loading
    data_set = pd.read_csv('training.csv', sep=';')
    test_data = pd.read_csv('validation.csv', sep=';')
    #
    df_merged = pd.concat([data_set, test_data], axis=1)

    # data preprocessing 
    # Drop Varible 18 for most Nans
    df_merged.drop(['variable18'], inplace=True, axis=1)
    # dropping ALL duplicte values
    df_merged.drop_duplicates(inplace=True)

    #Normlize variable17, variable14,variable15
    df_merged["variable17"] = df_merged["variable17"] / df_merged["variable17"].max()
    df_merged["variable14"] = df_merged["variable14"] / df_merged["variable14"].max()
    df_merged["variable15"] = df_merged["variable15"] / df_merged["variable15"].max()
    
    # Replace Nan values with the mean of each column
    df_merged['variable17'] = df_merged['variable17'].fillna(df_merged['variable17'].mean())
    df_merged['variable14'] = df_merged['variable14'].fillna(df_merged['variable14'].mean())
    df_merged['variable15'] = df_merged['variable15'].fillna(df_merged['variable15'].mean())
    
    # Drop all Nan rows (in charcters colums)
    df_merged = df_merged.dropna()
    #Extract labels 
    labels = df_merged.iloc[:, -1].values

    # Drop Labesls 
    df_merged.drop(['classLabel'], inplace=True, axis=1)
    df_merged = pd.get_dummies(df_merged)
    features = df_merged.iloc[:, :].values
    # Data Split
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
    #
    df_merged = pd.concat([data_set, test_data], axis=1)

    # data preprocessing 
    # Drop Varible 18 for most Nans
    df_merged.drop(['variable18'], inplace=True, axis=1)
    # dropping ALL duplicte values
    df_merged.drop_duplicates(inplace=True)

    #Normlize variable17, variable14,variable15
    df_merged["variable17"] = df_merged["variable17"] / df_merged["variable17"].max()
    df_merged["variable14"] = df_merged["variable14"] / df_merged["variable14"].max()
    df_merged["variable15"] = df_merged["variable15"] / df_merged["variable15"].max()
    
    # Replace Nan values with the mean of each column
    df_merged['variable17'] = df_merged['variable17'].fillna(df_merged['variable17'].mean())
    df_merged['variable14'] = df_merged['variable14'].fillna(df_merged['variable14'].mean())
    df_merged['variable15'] = df_merged['variable15'].fillna(df_merged['variable15'].mean())
    
    # Drop all Nan rows (in charcters colums)
    df_merged = df_merged.dropna()
    #Extract labels 
    labels = df_merged.iloc[:, -1].values

    # Drop Labesls 
    df_merged.drop(['classLabel'], inplace=True, axis=1)
    df_merged = pd.get_dummies(df_merged)
    features = df_merged.iloc[:, :].values
    # Data Split
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, shuffle=True)
    # model
    logreg_clf = LogisticRegression()
    # model fit
    model = logreg_clf.fit(x_train, y_train)
    # prediction
    predict = model.predict(data)
    return predict


model_train_testing_accuracy()
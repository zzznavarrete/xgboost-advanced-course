import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import (metrics, linear_model, preprocessing)
from sklearn.model_selection import train_test_split 
import xgboost as xgb 
# feature selection    make sure to update xgboost package
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, RepeatedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import uniform,randint


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

  
# Function importing Dataset 
def read_data(path): 
    df = pd.read_csv(path, na_values='NA')
    # Printing the dataswet shape 
    print ("features:") 
    print (list(df.columns)) 
    print ("row and column number") 
    print (df.shape) 
    print ("data / feature types:") 
    print (df.dtypes) 
    df_num = df.select_dtypes(include='number')
    df_cat = df.select_dtypes(include='object')
    print ("missing values:") 
    print (df.isnull().sum()) 
    return [df, df_num, df_cat] 

# data partition function 
def data_partition(df, features, target, seed): 
    X = df[features]
    Y = df[target]
  
    # partition data into training and testing data sets
    X_train, X_test, y_train, y_test = train_test_split(  
    X, Y, test_size = 0.33, random_state = seed) 
      
    return [X, Y, X_train, X_test, y_train, y_test] 



def run():
    path = "./../data/BankNote_Authentication.csv"
    dflist = read_data(path)

    # (0 for authentic, 1 for inauthentic)
    df = dflist[0]
    df.rename(columns = {'class':'target'}, inplace = True)
    df.head()
    df.target.mean()

    features = list(df.columns)
    features.remove('target')

    X, Y, X_train, X_test, y_train, y_test = data_partition(df, features, 'target', 12)

    it_times = 10
    tree_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 12, 
                max_depth = 3, min_samples_leaf = 5) 

    tree_entropy.fit(X_train, y_train) 

    prediction = tree_entropy.predict_proba(X_train)[:, 1]

    prediction_test = tree_entropy.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, prediction_test)  
    roc_auc = metrics.auc(fpr, tpr)  
    print ("auc at 0 step: ", roc_auc)

    err = y_train - prediction
    booster = [tree_entropy]

    for j in range(it_times):
        j = j + 1
        tree_model = DecisionTreeRegressor(criterion = 'squared_error', random_state = 12, 
                max_depth = 3, min_samples_leaf = 5) 
        
        tree_model.fit(X_train, err)     
        err_predict = tree_model.predict(X_train)
        prediction = prediction + err_predict
        err = err - err_predict
        booster.extend([tree_model])
        
        err_predict_test = tree_model.predict(X_test)
        prediction_test = prediction_test + 0.02*err_predict_test
        fpr, tpr, thresholds = metrics.roc_curve(y_test, prediction_test)  
        roc_auc = metrics.auc(fpr, tpr)  
        print ("auc at iteration " + str(j) + " step: ", roc_auc)
        
    


if __name__ == "__main__":
    run()

    
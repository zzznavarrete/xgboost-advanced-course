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
from matplotlib import pyplot
# feature selection    make sure to update xgboost package
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, RepeatedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import uniform,randint
from xgboost import plot_importance

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


path = '.\..\data\creditscore.csv'
dflist = read_data(path)

scorecard = dflist[0]
predictor = list(scorecard.columns)

# even though xgboost does not need missimg treatment, we still do that
# since we need to analyze data patterns, such as correlation..
scorecard['monthlyincome_miss']=pd.isnull(scorecard['monthlyincome'])+0 # Si le añado un 0 significa que el True/False se convierte en su equivalente binario 1/0
scorecard['numdependents_miss']=pd.isnull(scorecard['numdependents'])+0

# fill missing with mean value
scorecard = scorecard.fillna(scorecard.mean())


varnamelist=list(scorecard.columns)
varnamelist.remove('ID')
varnamelist.remove('overdue')
Y = scorecard.overdue
Y.mean()
Y.value_counts()

corr=list()   # define an empty list
# loop for all independent variables
for vname in varnamelist: 
        X = scorecard[vname]              # loop all the variables in the data frame
        C = np.corrcoef(X, Y)             # calculate correlation matrix
        beta=np.round(C[1, 0],3)
        corr=corr+[beta]                # gather all correlation coefficients in a list

# change column names
corrdf = pd.DataFrame({'varname': varnamelist, 'correlation': corr})
corrdf['abscorr'] = np.abs(corrdf['correlation'])

# sort absolute value of correlation in descending order
corrdf.sort_values(['abscorr'], ascending=False, inplace=True)
seq = range(1,len(corrdf)+1)
corrdf['order']=seq  # add a sequential number column

corrdf

# get one hot features for num30_59dayspastdue, num90dayslate and num60_89dayspastdue 
scorecard.num90dayslate.value_counts()

names = ['num30_59dayspastdue', 'num90dayslate', 'num60_89dayspastdue']
for it in names:
    df_1 = pd.get_dummies(scorecard[it])
    colnames = list(df_1.columns)
    colnames_new = [it + "_" + str(colname) for colname in colnames]
    df_1.columns = colnames_new
    scorecard = pd.concat([scorecard, df_1], axis=1)

list(scorecard.columns)

predictor = list(scorecard.columns)

for s in ['overdue', 'ID']:
    predictor.remove(s)
    
print (predictor)

X = scorecard[predictor]
y = scorecard.overdue

# Cross-validation
NFOLDS = 5
kf = KFold(n_splits = NFOLDS, shuffle = True)

for tr_idx, val_idx in kf.split(X, y):

    clf = xgb.XGBClassifier(
    learning_rate =0.03,
    n_estimators = 1200,
    max_depth = 7,
    objective= 'binary:logistic',
    min_child_weight=5, 
    gamma=0.05, 
    subsample=0.8, 
    colsample_bytree=0.8,
    seed = 12
  )
    
    X_tr, X_vl = X.iloc[tr_idx, :], X.iloc[val_idx, :]
    y_tr, y_vl = y.iloc[tr_idx], y.iloc[val_idx]
    clf.fit(X_tr, y_tr)
    y_pred_train = clf.predict_proba(X_vl)[:,1]
    
    print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))
    
    y_pred_train0 = clf.predict(X_vl)
    accuracy = accuracy_score(y_vl, y_pred_train0)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # if you predict new data, with the same features
    #y_preds += clf.predict_proba(X_new_sum_can)[:,1] / NFOLDS
    
    
      
    
    
def run():
    # now consider feature selection using importance level and  SelectFromModel
    # this method is applied only on sklearn xgboost rather than native booster
    X = scorecard[predictor]
    y = scorecard.overdue

    clf = xgb.XGBClassifier()
    clf.fit(X, y)  

    sorted_idx = np.argsort(clf.feature_importances_)
    important_varlist = [X.columns[index] for index in sorted_idx if clf.feature_importances_[index]>0]
    X = X[important_varlist]
    clf = xgb.XGBClassifier()
    clf.fit(X, y)  



    sorted_idx = np.argsort(clf.feature_importances_)
    important_varlist = [X.columns[index] for index in sorted_idx]
    thresholds = sorted(clf.feature_importances_)
    # or use the following
    #thresholds = [clf.feature_importances_[index] for index in sorted_idx]

    X_train, X_valid, y_train, y_valid = train_test_split( 
            X, y, test_size = 0.3, random_state = 81) 


    len(X_train.columns)

    # The threshold value to use for feature selection. Features 
    # whose importance is greater or equal are kept while the others are discarded


    #prefitbool, default False
    # Whether a prefit model is expected to be passed into the constructor directly 
    # or not. If True, transform must be called directly and SelectFromModel cannot 
    # be used with cross_val_score, GridSearchCV and similar utilities that clone 
    # the estimator. Otherwise train the model using fit and then transform to do 
    # feature selection.

    rg = range(len(thresholds))
    selected = important_varlist[:]
    for j in rg:
        thresh = thresholds[j]
        remvar = important_varlist[j]
        selection = SelectFromModel(clf, threshold = thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        selection_model = xgb.XGBClassifier()
        selection_model.fit(select_X_train, y_train)
        select_X_test = selection.transform(X_valid)
        print ('# of features:', str(select_X_test.shape[1]))
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_valid, predictions)
        print('accuracy: ',accuracy)
        predictions_prob = selection_model.predict_proba(select_X_test)[:,1]
        print('auc: ',roc_auc_score(y_valid, predictions_prob))
        selected.remove(remvar)
        X_train_select = X_train[selected]
        
        

    
    
    #############use native booster#####################################   

    X_train, X_valid, y_train, y_valid = train_test_split( 
            X, y, test_size = 0.15, random_state = 122) 


    params = {'learning_rate' : 0.03,
        'max_depth' : 7,
        'objective' : 'binary:logistic',
        'min_child_weight' : 5, 
        'gamma' : 0.05, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8,
        'seed' : 12,
        'eval_metric' : 'auc'
    }


    dtrain = xgb.DMatrix(data=X_train.values,
                        feature_names=X_train.columns,
                        label=y_train.values)

    dvalid = xgb.DMatrix(data=X_valid.values,
                        feature_names=X_valid.columns,
                        label=y_valid.values)

    #  'n_estimators' : 1200 in params is replaced by num_boost_round=1200

    # Here I use verbose_eval = 20 , this means the evaluation metric auc
    # is printed every 20 boosting stages, instead of every boosting stage
    # check https://xgboost.readthedocs.io/en/latest/python/python_api.html

    mod = xgb.train(params=params,
                    dtrain=dtrain,
                    num_boost_round = 1200,
                    early_stopping_rounds=50, 
                    evals=[(dvalid,'valid'), (dtrain,'train')],
                    verbose_eval=20)


    #######################use cv() to get best boosting round #####################

    X_train, X_valid, y_train, y_valid = train_test_split( 
            X, y, test_size = 0.27, random_state = 122) 


    dtrain = xgb.DMatrix(data=X_train.values,
                        feature_names=X_train.columns,
                        label=y_train.values)

    dvalid = xgb.DMatrix(data=X_valid.values,
                        feature_names=X_valid.columns,
                        label=y_valid.values)

    xgb_params = {'learning_rate' : 0.03,
        'max_depth' : 7,
        'objective' : 'binary:logistic',
        'min_child_weight' : 5, 
        'gamma' : 0.05, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8,
        'seed' : 12
    }

    eval_cv = xgb.cv(xgb_params, dtrain, num_boost_round=1200, nfold = 5,
                metrics='auc', early_stopping_rounds=80)
            
    clf = xgb.XGBClassifier(
        learning_rate =0.03,
        n_estimators = 1200,
        max_depth = 7,
        objective= 'binary:logistic',
        min_child_weight=5, 
        gamma=0.05, 
        subsample=0.8, 
        colsample_bytree=0.8,
        seed = 12)
        
        
    clf.set_params(n_estimators = eval_cv.shape[0])
            
    clf.fit(X_train, y_train, eval_metric='auc', verbose=True)
            
    predictions_prob = clf.predict_proba(X_valid)[:,1]      
        
    print('ROC AUC {}'.format(roc_auc_score(y_valid, predictions_prob)))


    #  we use the following ways
    imp_gain = clf.get_booster().get_score(importance_type="gain")
    imp_weight = clf.get_booster().get_score(importance_type="weight")


if __name__ == "__main__":
    run()
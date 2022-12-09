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


############## multi-classification problem:  'mlogloss' or "merror"#################################
path = r'C:\quick_course\xgb\data\trainforestcover.csv'
dflist = read_data(path)

df = dflist[0]
predictor = list(df.columns)

 # remove constant columns
remove = []
for col in df.columns:
    if df[col].std() == 0:
        remove.append(col)

df = df.drop(remove, axis=1)

# remove duplicated columns
remove = []
c = df.columns
for i in range(len(c)-1):
    v = df[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v, df[c[j]].values):
            remove.append(c[j])       
        
df = df.drop(remove, axis=1)

set(df['Cover_Type']) # 7 types

y = df['Cover_Type'] - 1

set(y)

X = df.drop(['Id','Cover_Type'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 19) 


dtrain = xgb.DMatrix(data=X_train,
                     feature_names=X_train.columns,
                     label=y_train)

dvalid = xgb.DMatrix(data=X_valid,
                     feature_names=X_valid.columns,
                     label=y_valid)

params = {'learning_rate' : 0.03,
    'max_depth' : 7,
    'objective': 'multi:softmax',
    'min_child_weight' : 5, 
    'gamma' : 0.05, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8,
    'seed' : 12,
    'num_class' : 7,
    'eval_metric' : 'mlogloss'  # can also be 'merror'
}

eval_cv = xgb.cv(params, dtrain,
  num_boost_round=1000, nfold=3, shuffle=True, verbose_eval=5, early_stopping_rounds=50)

eval_cv.columns
eval_cv.head()

# check log loss: -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
nround = eval_cv['test-mlogloss-mean'].idxmin()
# final model
model = xgb.train(params, dtrain, num_boost_round=nround)



# sklean XGBoostClassifier, we use eval_metric="merror"
y= df['Cover_Type'].values
y = [x-1 for x in y]
X = df.drop(['Id','Cover_Type'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 19) 


# classifier
clf = xgb.XGBClassifier(max_depth=7, 
   n_estimators=2500, learning_rate=0.02, nthread=4, subsample=0.9, 
   colsample_bytree=0.8, seed=9)

# fitting
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="merror", eval_set=[(X_valid, y_valid)])
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
%matplotlib inline  
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


path = r'C:\quick_course\xgb\data\train_liberty.csv'
dflist = read_data(path)

df = dflist[0]
predictor = list(df.columns)

df.Hazard.value_counts()

df_cat = df.select_dtypes(include='object')

cat_names = list(df_cat.columns)
for it in cat_names:
    df_1 = pd.get_dummies(df[it])
    colnames = list(df_1.columns)
    colnames_new = [it + "_" + str(colname) for colname in colnames]
    df_1.columns = colnames_new
    df = pd.concat([df, df_1], axis=1)

list(df.columns)

df.drop(cat_names, axis = 1, inplace = True)
predictor = list(df.columns)

df.dtypes
df.head(5)

for s in ['Id', 'Hazard']:
    predictor.remove(s)
    
print (predictor)

X = df[predictor]
y = df.Hazard

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.25, random_state=2)

dtrain = xgb.DMatrix(data=X_train,
                     feature_names=X_train.columns,
                     label=y_train)

dvalid = xgb.DMatrix(data=X_valid,
                     feature_names=X_valid.columns,
                     label=y_valid)

set(df.Hazard)
set(y_valid)

params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.01
params["min_child_weight"] = 5
params["subsample"] = 1
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 7

plst = list(params.items())

watchlist = [(dtrain, 'train'),(dvalid, 'val')]
model = xgb.train(plst, dtrain, 500, watchlist, early_stopping_rounds=5)



# plot feature importance
plot_importance(model, importance_type='gain')
pyplot.show()

#  we use the following ways
imp_gain = model.get_score(importance_type="gain")
imp_weight = model.get_score(importance_type="weight")

important_values = list(imp_gain.values())
important_vars = list(imp_gain.keys())
sorted_idx = np.argsort(important_values)[::-1]
important_var_gain = [(important_vars[index], important_values[index]) for index in sorted_idx]

important_values = list(imp_weight.values())
important_vars = list(imp_weight.keys())
sorted_idx = np.argsort(important_values)[::-1]
important_var_weight = [(important_vars[index], important_values[index]) for index in sorted_idx]

import matplotlib.pyplot as plt
import seaborn as sns
feature_imp = pd.DataFrame(important_var_gain, columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
plt.bar(feature_imp.Feature, feature_imp.Value)
plt.show()

# other way seaborn for importance levels
pic=sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
pic.figure.savefig('importance.png')


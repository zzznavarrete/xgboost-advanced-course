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

path = r'C:\quick_course\xgb\data\Trainingonline.csv'
df = pd.read_csv(path, na_values='NA')
print (list(df.columns))
df.head(6)

# we want to check how many missing values for each column
missing = df.isnull().sum()

# from 12th element in array (i.e. 13th column) are model label features
print (missing[12:20])

# give missing # for all features, and those feature names
# varlist is a list containing the names (str) that have missing values
missingnew = missing[12:]
withmissing = missingnew[missing>0]
varlist = list(withmissing.index)

# create missing dummy for each column with missing values
for varname in varlist:
        missdummy = str(varname)+'_missdummy'
        df[missdummy] = pd.isnull(df[str(varname)])+0

# perform missing value imputation   
for varname in varlist:
      df[str(varname)].fillna(df[str(varname)].median(), inplace=True)


# we prove that Date_1>Date_2
print (np.min(df.Date_1-df.Date_2))
print (np.max(df.Date_1-df.Date_2))
print (np.mean(df.Date_1-df.Date_2))


# The shape of original data frame is (751, 584) (584 features), 
# but the first 12 columns are are dependents variables (model targets)
# So we should transform the first 12 columns into 12 rows: one row -> 12 rowa
# also we will use month as a predictor.


Xlist=[]
Ylist=[]
# To create the data frame we want, we iterate over rows of the dataframe
# create X matrix target variable, they now become np array
# at the same time, the month of the year is added as a predictor
m =[]
for index, row in enumerate(df.values):
     for j in range(0,12):
            if not np.isnan(row[j]): 
                Ylist.append(row[j])   
                r = list(row[12:572])
                month = [j+1]
                r.extend(month)
                vlist = np.array(r)
                Xlist.append(vlist)  # append a month of year
                
X = np.array(Xlist)
y = np.log(np.array(Ylist)+1)

# show matrix for ind and dep vars
print (X.shape)
print (y.shape)
       
# make binary variables from categorical variable
# these are the colunms after we remove the first 12 target columns

names=list(df.columns.values)[12:558]

# get the list categorical features only containing 
# the 'Cat_', categorical variables
cat = []
for j in range(0,len(names)):
    if "Cat_" in names[j]: 
         cat.extend([j])

# the # of categorical features            
print (len(cat))

# make binary variables from categorical variable
# we first use label encoder

from sklearn import (metrics, linear_model, preprocessing)

X_cp = X.copy()

for i in cat:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(X_cp[:,i])
        s = encoder.transform(X_cp[:,i])    
        array_category = np.array(pd.get_dummies(s))
        if i==cat[0]:
             X_cp_cat = array_category.copy()
        else:                  
             X_cp_cat = np.concatenate([X_cp_cat, array_category], axis=1)

# remove cat variables        
for i in cat:
        np.delete(X_cp, np.s_[i], 1)
        
X_new = np.concatenate([X_cp, X_cp_cat], axis=1)
 
print ('X dimension', X_new.shape)
print ('Y dimension', y.shape)

L = X_new.shape[1]
v = ['var_' + str(j) for j in range(L)]
X_new_df = pd.DataFrame(X_new, columns = v )
X_new_df.columns
X_new_df.shape


reg = xgb.XGBRegressor()
reg.fit(X_new_df, y)  

# get feature importance and ranking
important_values = reg.feature_importances_
sorted_idx = np.argsort(important_values)[::-1]

# remove features with importance = 0
important_var_gain = [(X_new_df.columns[index], important_values[index]) for index in sorted_idx if important_values[index] > 0]
# we have 400 features with importance value>0
len(important_var_gain)

# restructure data and refit
important_varlist = [it[0] for it in important_var_gain]
X_new_df = X_new_df[important_varlist]
X_new_df.shape

reg = xgb.XGBRegressor()
reg.fit(X_new_df, y) 

important_values = reg.feature_importances_
sorted_idx = np.argsort(important_values)[::-1]
important_var_gain = [(X_new_df.columns[index], important_values[index]) for index in sorted_idx]




# Theshold: If None and if the estimator has a parameter penalty set to l1,
# either explicitly or implicitly (e.g, Lasso), 
# the threshold used is 1e-5. Otherwise, “mean” is used by default.
modelselect = SelectFromModel(reg, prefit=True)
X_new = modelselect.transform(X_new_df)
X_new.shape

# we'd better use data frame
chosen = modelselect.get_support()
chosen_names = [list(X_new_df.columns)[j] for j in range(len(chosen)) if chosen[j] == True]
len(chosen_names)
X = X_new_df[chosen_names]
list(X.columns)
X.shape

####################fit xgboost regressor#######################

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 19) 

xgbreg = xgb.XGBRegressor(
    learning_rate =0.02,
    n_estimators = 2500,
    max_depth = 7,
    objective= 'reg:squarederror',
    min_child_weight=5, 
    gamma=0.05, 
    subsample=0.8, 
    colsample_bytree=0.8,
    seed = 9)

# fitting: using early_stopping_rounds we get the best n_estimators
xgbreg.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="rmse", eval_set=[(X_valid, y_valid)])

# we get 1450 iteration time
# Other parameter tuning

# RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, KFold
from scipy.stats import uniform,randint

xgbreg = xgb.XGBRegressor(objective = 'reg:squarederror')
param_dist = {'n_estimators': randint(1000, 2000),
              'learning_rate': uniform(0.01, 0.06),
              'subsample': [0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
              'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
              'gamma' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
              'colsample_bytree': [0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
              'min_child_weight': [1, 2, 3, 5, 7]
             }

# verbose: integer
# Controls the verbosity: the higher, the more messages.

regcv = RandomizedSearchCV(xgbreg, param_distributions = param_dist, 
    n_iter = 25, scoring = 'neg_mean_squared_error', cv = 3,
    error_score = 0, verbose = 3, n_jobs = -1)

search = regcv.fit(X_train, y_train)

search.best_params_

'''
{'colsample_bytree': 0.8,
 'gamma': 0.2,
 'learning_rate': 0.014930472072121975,
 'max_depth': 9,
 'min_child_weight': 2,
 'n_estimators': 1635,
 'subsample': 0.85}

'''

# Bayesian optimization
# http://krasserm.github.io/2018/03/21/bayesian-optimization/
# https://www.kdnuggets.com/2019/07/xgboost-random-forest-bayesian-optimisation.html
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error


dtrain = xgb.DMatrix(X_train, label=y_train)

def xgb_evaluate(max_depth, gamma, colsample_bytree, eta, n_estimators):
    params = {'eval_metric': 'rmse',
              'max_depth': int(max_depth),
              'subsample': 0.8,
              'n_estimators': int(n_estimators),
              'eta': eta,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]


xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 10), 
                                             'gamma': (0, 1),
                                             'eta': (0.01, 0.2),
                                             'n_estimators': (1000, 2000),
                                             'colsample_bytree': (0.4, 0.95)})
    
# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
# X: Points at which EI shall be computed     
xgb_bo.maximize(init_points=5, n_iter=25, acq='ei')  # ei means expected improvement

# Extract the parameters of the best model.


paramslist = pd.DataFrame(xgb_bo.res).sort_values(['target']).params[len(xgb_bo.res) -1]

# then set new parameter
params['max_depth'] = int(paramslist['max_depth'])
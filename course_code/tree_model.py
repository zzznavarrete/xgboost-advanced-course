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
import ipdb; 


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



      
# get tree model here we use gini measure to split nodes
def tree_model(X_train, y_train, criterionv, random_statev, max_depthv, min_samples_leafv): 
  
    clf = DecisionTreeClassifier(criterion = criterionv, 
      random_state = random_statev, max_depth = max_depthv,
      min_samples_leaf = min_samples_leafv)
  
    # fit decision tree -- obtain tree model
    clf.fit(X_train, y_train) 
    return clf 
      
  
# perform predictions using decision tree model
def tree_prediction(X_test, clf): 
    y_pred = clf.predict(X_test) 
    print(y_pred) 
    return y_pred 
      
# obtaining accuracy 
def get_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ", 
        confusion_matrix(y_test, y_pred)) 
      
    print ("Accuracy : ", 
    accuracy_score(y_test,y_pred)*100) 
      
    print("Report : ", 
    classification_report(y_test, y_pred)) 

def plot_confusion(labels, y_test, pred):  
    cm = confusion_matrix(y_true=y_test, y_pred=pred, labels=labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


    
def run():
    #ipdb.set_trace()
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


    tree_gini = tree_model(X_train, y_train, 'gini', 25, 3, 4)
    tree_entropy = tree_model(X_train, y_train, 'entropy', 67, 3, 4)

    gini_predict= tree_prediction(X_test, tree_gini) 
    
    get_accuracy(y_test, gini_predict) 
    plot_confusion([0,1], y_test, gini_predict)
    
    entropy_predict= tree_prediction(X_test, tree_entropy) 
    get_accuracy(y_test, entropy_predict)     
    
    plot_confusion([0,1], y_test, entropy_predict)  


    # tree plot
    from sklearn import tree
    tree.plot_tree(tree_gini)

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    tree.plot_tree(tree_gini,feature_names = features,class_names= ['0', '1'],filled = True)
    # save to png file          
    fig.savefig('./gini_tree.png')

    # AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_test, entropy_predict)  
    roc_auc = metrics.auc(fpr, tpr)  
    print ("auc: roc prob -- auc", roc_auc)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, gini_predict)  
    roc_auc = metrics.auc(fpr, tpr)  
    print ("auc: roc prob -- auc", roc_auc)



if __name__ == "__main__":
    run()
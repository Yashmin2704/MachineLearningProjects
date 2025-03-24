import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

from mypipes import *

import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt



import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.6f} (std: {1:.6f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    score = roc_auc_score(true_y, y_prob)
    print(f"ROC AUC: {score:.4f}")



#Logistic Regression
def LogisticR(x_train,y_train,x_test,y_test):
    params={'class_weight':['balanced',None],'penalty':['l1','l2'],'C':[.0001,.0005,.001,.005,.01,.05,.1,1,2,5]}
    model=LogisticRegression(fit_intercept=True)
    '''grid_search=GridSearchCV(model, param_grid=params,cv=10,
                         scoring="roc_auc",
                         n_jobs=-1,
                         verbose=20)
    grid_search.fit(x_train,y_train)
    logr=grid_search.best_estimator_
    print(report(grid_search.cv_results_,5))'''


    logr=LogisticRegression(fit_intercept=True,
                            **{'C': 0.0001, 'class_weight': 'balanced', 'penalty': 'l2'},solver='liblinear')
    # default solver lbfgs does not support l1 penalty for some versions of sklearn
    # if you get an error like that , simply use solver='liblinear', it supports both l1 & l2 penalty


    logr.fit(x_train,y_train)

    print(logr.intercept_)

    a=list(zip(x_train.columns,logr.coef_[0]))
    print(a)

    print(logr.predict_proba(x_train))

    

    cutoffs=np.linspace(0.01,0.99,99)

    

    logr.predict_proba(x_train)


    train_score=logr.predict_proba(x_train)[:,1]
    real=y_train
    print("train score")
    print(train_score)
    print("real")
    print(real)
     # In  order to find the probability of which column is for outcome 1 and which for outcome 0


    (train_score>0.2).astype(int)

    KS_all=[]
    Acc_all=[]
    Recall_all=[]
    Precission_all=[]
    F1_all=[]
    Sp_all=[]

    for cutoff in cutoffs:

        predicted=(train_score>cutoff).astype(int)
        
        if isinstance(real, pd.DataFrame):
            real = real.iloc[:, 0]

        TP=((predicted==1) & (real==1)).sum()
        TN=((predicted==0) & (real==0)).sum()
        FP=((predicted==1) & (real==0)).sum()
        FN=((predicted==0) & (real==1)).sum()

        P=TP+FN
        N=TN+FP


        KS=(TP/P)-(FP/N)
        Acc=(TP+TN)/(P+N)
        Recall=TP/(TP+FN)
        pre=TP/(TP+FP)
        F1=(2*Recall*pre)/(Recall+pre)
        sp=TN/(FP+TN) #specification

        KS_all.append(KS)
        Acc_all.append(Acc)
        Recall_all.append(Recall)
        Precission_all.append(pre)
        F1_all.append(F1)
        Sp_all.append(sp)




    list(zip(cutoffs,KS_all))

    mycutoff=cutoffs[KS_all==max(KS_all)]
    print(pd.DataFrame({'Cut-Off': cutoffs,
              'F1':F1_all, 
              'Precision':Precission_all,
              'Recall':Recall_all,
             'Acc': Acc_all,
             'KS':KS_all}))
    print("beta0:",logr.intercept_)
    b=list(zip(x_train.columns,logr.coef_[0]))
    print(b)
    print(logr.predict_proba(x_test))
    test_score=logr.predict_proba(x_test)[:,1]
    print(test_score)
    test_classes=(test_score>mycutoff).astype(int)
    print(roc_curve(y_test, test_score))
    print(plot_roc_curve(y_test, test_score))
    return(test_classes)


#decision tree
def decision(x_train,y_train,x_test,y_test):

    params={'class_weight':[None,'balanced'], 
            'max_depth':[None,5,10,15,20,30,50,70],
            'min_samples_leaf':[1,2,5,10,15,20], 
            'min_samples_split':[2,5,10,15,20]
           }

    clf=DecisionTreeClassifier()

    #random_search=RandomizedSearchCV(clf,cv=5,param_distributions=params,scoring='f1',n_iter=5,n_jobs=-1,verbose=20)
    #random_search.fit(x_train,y_train)
    #print(report(random_search.cv_results_,5))

    dtree=DecisionTreeClassifier(**{'min_samples_split': 20, 
                                    'min_samples_leaf': 10, 
                                    'max_depth': 10, 
                                    'class_weight': 'balanced'})

    dtree_fit =dtree.fit(x_train,y_train)

    test_score=dtree_fit.predict_proba(x_test)[:,1]
    

    y_pred = dtree_fit.predict(x_test)
        
    print(classification_report(y_test, y_pred, target_names=['0', '1']))

   # print(pd.crosstab(y_test, y_pred))
    
    print(roc_curve(y_test, test_score))
    print(plot_roc_curve(y_test, test_score))

    #dtree_fit.predict(pd.DataFrame(x_test.iloc[[5]]))
    return(test_score)
    
    
    
    
def random(x_train,y_train,x_test,y_test):
    clf = RandomForestClassifier()

    param_dist = {"n_estimators":[100,200,300,500,700,1000],
              "max_features": [5,10,20,25,30,35],
              "bootstrap": [True, False],
              'class_weight':[None,'balanced'], 
              'max_depth':[None,5,10,15,20,30,50,70],
              'min_samples_leaf':[1,2,5,10,15,20], 
              'min_samples_split':[2,5,10,15,20]
                  }
    
    random_search = RandomizedSearchCV(clf, 
                                       param_distributions=param_dist,
                                       n_iter=5,
                                       scoring='f1',
                                       cv=5,
                                       n_jobs=-1,
                                       verbose=20)
    '''random_search.fit(x_train, y_train)

    print(random_search.best_estimator_)

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=50, max_features=10, max_leaf_nodes=None,
            min_impurity_decrease=1e-07, min_samples_leaf=10,
            min_samples_split=20, min_weight_fraction_leaf=0.0,
            n_estimators=300, n_jobs=1, oob_score=False, random_state=None,
            verbose=0, warm_start=False)


    report(random_search.cv_results_,3)'''
    
    # select the best values from results above, they will vary slightly with each run
    rf = RandomForestClassifier(**{'n_estimators': 500, 
                               'min_samples_split': 20, 
                               'min_samples_leaf': 10, 
                               'max_features': 10, 
                               'max_depth': 50, 
                               'class_weight': 'balanced', 
                               'bootstrap': True})
    random_search.fit(x_train,y_train)

    y_pred_rf = random_search.predict(x_test)

    print(classification_report(y_test, y_pred_rf, target_names=['0', '1']))

    #print(pd.crosstab(y_test, y_pred_rf))
    feat_imp_df=pd.DataFrame({'features':x_train.columns,
                          'importance':rf.feature_importances_})

    feat_imp_df=feat_imp_df.sort_values('importance',ascending=False)
    feat_imp_df['cum_imp']=np.cumsum(feat_imp_df['importance'])
    x=feat_imp_df.reset_index()
    print(x_train[x[:500]['features']])
    return(y_pred_rf,x)

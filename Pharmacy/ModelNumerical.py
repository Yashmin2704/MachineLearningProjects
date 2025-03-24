import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

from mypipes import *

import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV



def LinearR (x_train,y_train,x_test,y_test):
    lm=LinearRegression()
    cv_mae=-cross_val_score(lm,x_train,y_train,cv=5,scoring='neg_root_mean_squared_error')

    print("cross validation mean:",cv_mae.mean())
    print("cross validation std:",cv_mae.std())
    lm.fit(x_train,y_train)
    print("beta0:",lm.intercept_)
    #print(lm.coef_)
    test_pred=lm.predict(x_test)
    df=pd.DataFrame({'Act':y_test, 'Pred':test_pred})
    print(df)
    #rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    absolute_errors = np.abs(y_test-test_pred)
    # Calculate the mean of the absolute errors
    mae = np.mean(absolute_errors)

    return(mae,test_pred)
    #return(rmse,test_pred)

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.6f} (std: {1:.6f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    
def RidgeR(x_train,y_train,x_test,y_test):
    lambdas=np.linspace(1,100,100)
    params={'alpha':lambdas}
    model=Ridge()
    grid_search=GridSearchCV(model,param_grid=params,cv=5,scoring='neg_root_mean_squared_error', verbose=20,n_jobs=-1)
    grid_search.fit(x_train,y_train)
    Ridge_model=grid_search.best_estimator_
    #grid_search.cv_results_
    print(report(grid_search.cv_results_,5))
    Ridge_model.fit(x_train,y_train)
    test_pred=Ridge_model.predict(x_test)
    #rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    #return(rmse,test_pred)
    absolute_errors = np.abs(y_test-test_pred)
    # Calculate the mean of the absolute errors
    mae = np.mean(absolute_errors)

    return(mae,test_pred)

def LassoR(x_train,y_train,x_test,y_test):
    
    lambdas=np.linspace(1,10,100)
    model=Lasso(fit_intercept=True)
    params={'alpha':lambdas}
    grid_search=GridSearchCV(model,param_grid=params,cv=10,scoring='neg_mean_absolute_error',verbose=20,n_jobs=-1)
    grid_search.fit(x_train,y_train)
    lasso_model=grid_search.best_estimator_
    #grid_search.cv_results_
    print(report(grid_search.cv_results_,5))
    lasso_model.fit(x_train,y_train)
    print(lasso_model.intercept_)
    print(lasso_model.coef_)
    test_pred=lasso_model.predict(x_test)
    #rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    #return(rmse,test_pred)
    absolute_errors = np.abs(y_test-test_pred)
    # Calculate the mean of the absolute errors
    mae = np.mean(absolute_errors)

    return(mae,test_pred)


def decision(x_train,y_train,x_test,y_test):
    params={ 
            'max_depth':[None,5,10,15,20,30,50,70],
            'min_samples_leaf':[1,2,5,10,15,20], 
            'min_samples_split':[2,5,10,15,20]
           }

    reg=DecisionTreeRegressor()

    random_search=RandomizedSearchCV(reg,
                                     cv=5,
                                     param_distributions=params,
                                     scoring='neg_mean_absolute_error',
                                     n_iter=60,
                                     n_jobs=-1,verbose=20
                                        )

    random_search.fit(x_train,y_train)

    print(report(random_search.cv_results_,5))

    print("best estimator",random_search.best_estimator_)

    '''dt_reg = DecisionTreeRegressor(**{'min_samples_split':20, 
                                      'min_samples_leaf': 15, 
                                      'max_depth': 10})

    dt_reg.fit(x_train, y_train)'''

    #test_pred=dt_reg.predict(x_test)
    test_pred=random_search.predict(x_test)

    #rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    #return(rmse,test_pred)
    absolute_errors = np.abs(y_test-test_pred)
    # Calculate the mean of the absolute errors
    mae = np.mean(absolute_errors)

    return(mae,test_pred)


def random(x_train,y_train,x_test,y_test):
    param_dist = {"n_estimators":[50,100,200],
                  "max_features": [2,4,5,6,8],
                  "bootstrap": [True, False],
                  'max_depth':[None,5,10,15,20,30,50,70],
                  'min_samples_leaf':[1,2,5,10,15,20], 
                  'min_samples_split':[2,5,10,15,20]
                      }

    reg=RandomForestRegressor()

    random_search=RandomizedSearchCV(reg,
                                     cv=5,
                                     param_distributions=param_dist,
                                     scoring='neg_mean_absolute_error',
                                     n_iter=80,n_jobs=-1,verbose=20
                                        )

    random_search.fit(x_train,y_train)

    print(report(random_search.cv_results_,5))
    print(random_search.best_estimator_)
    #bootstrap=False, max_features=5, min_samples_split=15,n_estimators=200

    '''rf_reg = RandomForestRegressor(**{'n_estimators': 200, 
                                      'min_samples_split': 15, 
                                      'min_samples_leaf': 2, 
                                      'max_features': 5, 
                                      'max_depth': 5, 
                                      'bootstrap': False})

    rf_reg.fit(x_train, y_train)

    rf_test_pred=rf_reg.predict(x_test)'''
    rf_test_pred=random_search.predict(x_test)

    #rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    #return(rmse,rf_test_pred)
    
    absolute_errors = np.abs(y_test-rf_test_pred)
    # Calculate the mean of the absolute errors
    mae = np.mean(absolute_errors)

    return(mae,rf_test_pred)





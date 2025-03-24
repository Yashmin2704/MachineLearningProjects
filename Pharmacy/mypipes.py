import pandas as pd
import numpy as np
import re

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


class VarSelector(BaseEstimator, TransformerMixin):

    def __init__(self,feature_names):

        self.feature_names=feature_names


    def fit(self,x,y=None):

        return self

    def transform(self,X):

        return X[self.feature_names]

    def get_feature_names(self):

        return self.feature_names
    
    def get_feature_names_out(self, feature_names_out):

        return self.feature_names


class custom_fico(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.feature_names=['fico']

    def fit(self,x,y=None):

        return self

    def transform(self,X):

        k=X['FICO.Range'].str.split('-',expand=True).astype(float)
        fico=0.5*(k[0]+k[1])
        return pd.DataFrame({'fico':fico})

    def get_feature_names(self):

        return self.feature_names
    
    def get_feature_names_out(self, feature_names_out):

        return self.feature_names


class custom_age_band(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.feature_names=['age_band']

    def fit(self,x,y=None):

        return self

    def transform(self,X):

        k=X['age_band'].str.split('-',expand=True)
        k[0]=pd.to_numeric(k[0],errors='coerce')
        k[1]=pd.to_numeric(k[1],errors='coerce')
        age_band=0.5*(k[0]+k[1])
        age_band=np.where(X['age_band'].str[:2]=='71',71,age_band)
        return pd.DataFrame({'age_band':age_band})

    def get_feature_names(self):

        return self.feature_names
    
    def get_feature_names_out(self, feature_names_out):

        return self.feature_names


class custom_family_income(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.feature_names=['fi']

    def fit(self,x,y=None):

        return self

    def transform(self,X):

        k=X['family_income'].str.replace(',','')
        k=k.str.replace('<','')
        k=k.str.replace('>=','')
        k=k.str.strip()
        k=k.str.replace('  ',' ')
        a=k.str.split(' ',expand=True)

        a[0]=pd.to_numeric(a[0],errors='coerce')
        a[1]=pd.to_numeric(a[1],errors='coerce')
        fi=0.5*(a[0]+a[1])

        fi=np.where(k=='35000',35000,fi)
        fi=np.where(k=='4000',4000,fi)

        return pd.DataFrame({'fi':fi})

    def get_feature_names(self):

        return self.feature_names
    
    def get_feature_names_out(self, feature_names_out):

        return self.feature_names


        

class string_clean(BaseEstimator, TransformerMixin):

    def __init__(self,replace_it='',replace_with=''):

        self.replace_it=replace_it
        self.replace_with=replace_with
        self.feature_names=[]

    def fit(self,x,y=None):

        self.feature_names=x.columns
        return self

    def transform(self,X):

        for col in X.columns:
            X[col]=X[col].str.replace(self.replace_it,self.replace_with)
        return X
    def get_feature_names(self):

        return self.feature_names
    
    def get_feature_names_out(self, feature_names_out):

        return self.feature_names




class convert_to_numeric(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.feature_names=[]

    def fit(self,x,y=None):
        self.feature_names=x.columns
        return self

    def transform(self,X):
        for col in X.columns:
            X[col]=pd.to_numeric(X[col],errors='coerce')
        return X
    def get_feature_names(self):
        return self.feature_names
     
    def get_feature_names_out(self, feature_names_out):
        return self.feature_names


class get_dummies_Pipe(BaseEstimator, TransformerMixin):

    def __init__(self,freq_cutoff=0):

        self.freq_cutoff=freq_cutoff
        self.var_cat_dict={}
        self.feature_names=[]

    def fit(self,x,y=None):

        data_cols=x.columns

        for col in data_cols:

            k=x[col].value_counts()

            if (k<=self.freq_cutoff).sum()==0:
                cats=k.index[:-1]

            else:
                cats=k.index[k>self.freq_cutoff]

            self.var_cat_dict[col]=cats

        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                self.feature_names.append(col+'_'+cat)
        return self

    def transform(self,x,y=None):
        dummy_data=x.copy()

        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                name=col+'_'+cat
                dummy_data[name]=(dummy_data[col]==cat).astype(int)

            del dummy_data[col]
        return dummy_data

    def get_feature_names(self):

        return self.feature_names
    
    def get_feature_names_out(self, feature_names_out):

        return self.feature_names

class DataFrameImputer(BaseEstimator,TransformerMixin):

    def __init__(self):

        self.impute_dict={}
        self.feature_names=[]

    def fit(self, X, y=None):

        self.feature_names=X.columns

        for col in X.columns:
            if X[col].dtype=='O':
                self.impute_dict[col]='missing'
            else:
                self.impute_dict[col]=X[col].median()
        return self

    def transform(self, X, y=None):
        return X.fillna(self.impute_dict)

    def get_feature_names(self):

        return self.feature_names
    
    def get_feature_names_out(self, feature_names_out):

        return self.feature_names

class pdPipeline(Pipeline):

    def get_feature_names(self):

        last_step = self.steps[-1][-1]

        return last_step.get_feature_names()

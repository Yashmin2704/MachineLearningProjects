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
                self.feature_names.append(col+'_'+str(cat))
        return self

    def transform(self,x,y=None):
        dummy_data=x.copy()

        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                name=col+'_'+str(cat)
                dummy_data[name]=(dummy_data[col]==cat).astype(int)

            del dummy_data[col]
        return dummy_data

    def get_feature_names(self):

        return self.feature_names
    
    def get_feature_names_out(self, feature_names_out):

        return self.feature_names

class Date_data(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = ['day_sin', 'day_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()
        # Convert the 'datetime' column to datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Extract the date
        df['date'] = df['datetime'].dt.date
        df['ListDate'] = pd.to_datetime(df['ListDate'])

        # Extract date components
        df['day'] = df['ListDate'].dt.day
        df['month'] = df['ListDate'].dt.month
        df['weekday'] = df['ListDate'].dt.dayofweek

        # Normalize components
        df['day_normalized'] = df['day'] / 31.0
        df['month_normalized'] = df['month'] / 12.0
        df['weekday_normalized'] = df['weekday'] / 7.0

        # Apply sine and cosine transformations
        df['day_sin'] = np.sin(2 * np.pi * df['day_normalized'])
        df['day_cos'] = np.cos(2 * np.pi * df['day_normalized'])
        df['month_sin'] = np.sin(2 * np.pi * df['month_normalized'])
        df['month_cos'] = np.cos(2 * np.pi * df['month_normalized'])
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday_normalized'])
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday_normalized'])

        df_transformed = df[self.feature_names]
        return df_transformed

    def get_feature_names_out(self, input_features=None):
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

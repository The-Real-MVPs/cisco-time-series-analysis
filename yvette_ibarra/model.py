import pandas as pd
import wrangle as wr
import summaries as s
from importlib import reload
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import statsmodels.api as sm

import warnings
warnings.filterwarnings('ignore')

import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

def modeling_preprocessing(df:pd.DataFrame, columns_to_keep=['customer_type', 'month_name','day_name', 'quarter'], regression:bool=True, np_array:bool=True):
    '''
    prepares data for the modeling
    Warning! Try not to change np_array default setting. Creating dummies for data frame is very risky! 
    The length of the train and test sets with dummies might be different
    
    Parameters:
        df: clean data frame
        regression: 
            True if regression modeling
            False if time series modeling
        np_array:
            True: return np.array of arrays with OneHotEncoding for X_train and X_test
            False: return data frames for X_train and X_test
    
    Returns:
        X_train, y_train, X_test, y_test
        
    '''
    df = prepare_df_preprocessing(df, columns_to_keep)
    train, test = wr.split_data(df)

    X_train, y_train, X_test, y_test = train.iloc[:, :-1], train.iloc[:,-1], test.iloc[:, :-1], test.iloc[:,-1]
    # change data types
    for col in X_train.columns:
        X_train[col] = pd.Categorical(X_train[col])
        X_test[col] = pd.Categorical(X_test[col])
    if regression:
        # regression time series
            if np_array:
                ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse=False).fit(X_train)
                X_train = ohe.transform(X_train)
                X_test = ohe.transform(X_test)
            else:
                X_train = create_dummies(X_train)
                X_test = create_dummies(X_test)
    else:
        # pure time series
        X_train = train.purchase_amount
        X_test = test.purchase_amount

    
    return X_train, y_train, X_test, y_test

def create_dummies(df):
    '''
    create dummy variables for all categorical columns
    this function might not work for modeling, as the test data set might not get all date features
    in this case use the parameter np_array = True in the preprocessing function
    
    Parameters:
        df -> train or test data set
    Return:
        df with encoded categorical varaibles
    '''
    dummies_q = pd.get_dummies(df.quarter, drop_first=True)
    dummies_m = pd.get_dummies(df.month, drop_first=True)
    dummies_w = pd.get_dummies(df.week, drop_first=True)
    dummies_d = pd.get_dummies(df.day_of_week, drop_first=True)
    df['is_school'] = np.where(df.customer_type == 'K-12', 1, 0)
    df['is_gov'] = np.where(df.customer_type == 'Local Goverment', 1, 0)
    df['is_edu'] = np.where(df.customer_type == 'Higher Ed', 1, 0)
    df['is_state'] = np.where(df.customer_type == 'State Agency', 1, 0)
    df = df[['is_school', 'is_gov', 'is_edu', 'is_state']]
    return pd.concat([df, dummies_d, dummies_m, dummies_w, dummies_q],axis=1)
    
def prepare_df_preprocessing(df:pd.DataFrame, columns_to_keep:list, target='purchase_amount') -> pd.DataFrame:
    '''
    Keep only needed columns
    Convert them to category type
    Parameters: 
        df: dataframe before splitting
    Return: 
        df: ready to split
    '''
    
    # change data types
    for col in columns_to_keep:
        df[col] = pd.Categorical(df[col])
        df[col] = pd.Categorical(df[col])
    columns_to_keep.append(target)
    df = df[columns_to_keep]
    return df
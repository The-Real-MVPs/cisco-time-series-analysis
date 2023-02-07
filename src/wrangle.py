import pandas as pd
import numpy as np
import os

########### GLOBAL VARIABLES ###########
start2018_ = False
######## ACQUIRE FUNCTIONS #############

def get_start2018():
    
    return start2018_

def acquire_data() -> pd.DataFrame:
    '''
    Reads the data from the csv file. 
    Filters data by the Vendor Name (Cisco)
    Renames columns by making them lower case replacing white space with underscore
    Removes columns where all values are NULL
    Saves the data into data.pickle file.
    Next interation reads the data from the saved file

    Returns: pandas DataFrame
    '''
    filename = 'data/data.pickle'
    filename_csv = 'data/OFFICIAL_DIR_Cooperative_Contract_Sales_Data___Fiscal_2010_To_Present.csv'
    if os.path.isfile(filename):
        # read the filtered data  
        df = pd.read_pickle(filename)
        return df
    # if not available, go to the file downloaded from https://data.texas.gov/
    # the full link to the site available in the Readme file
    else:
        try:
            # read the csv file with the data of all companies
            df1 = pd.read_csv(filename_csv, low_memory=False)
            # filter by Vendor Name is Cisco 
            df = df1[df1["Vendor Name"].str.contains('Cisco')].copy()
            # rename columns into programming friendly format
            # to lower case, white spaces replaced by underscore
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            # drop the columns where all values are NULL
            df.drop(['staffing_contractor_name',
                'staffing_technology', 'staffing_title', 'staffing_level',
                'staffing_technology_type', 'staffing_start_date',
                'staffing_acquistion_type'], axis=1, inplace=True)
            # save the file
            pd.to_pickle(df, filepath_or_buffer=filename)
            return df
        except FileNotFoundError:
            # if file not found print:
            print('The file doesn\'t exist. Please, download it from the link provided in the Readme file')  

def basic_clean(df:pd.DataFrame, start2018=start2018_) -> pd.DataFrame:
    '''
    Remove unneeded columns
    Create a copy of order date
    Changes dates type
    Sets order date as an index
    Parameters:
        df: pandas data frame
    Returns:
        cleaned data frame
    '''
    # list of columns to drop
    drop_columns = ['fiscal_year',
     'rfo_description',
     'rfo_number',
     'contract_number',
     'customer_contact',
     'customer_address',
     'customer_state',
     # 'customer_zip',
     'vendor_name',
     'vendor_contact',
     'vendor_hub_type',
     'vendor_address',
     'vendor_state',
     'vendor_city',
     'vendor_zip',
     'reseller_hub_type',
     'reseller_address',
     'reseller_state',
     'reseller_zip',
     'reseller_phone',
     'report_received_month',
     'brand_name',
     'purchase_month',
     'invoice_number',
     'dir_contract_mgr',
     'contract_type',
     'contract_subtype',
     'contract_start_date',
     'contract_end_date',
     'contract_termination_date',
     'sales_fact_number']

    df = df.drop(columns = drop_columns, axis=1)
    # create a copy for the shipped date
    df['order_date_copy'] = df.order_date

    # convert order date and shpping date to datetime
    df.order_date = pd.to_datetime(df.order_date)
    df.shipped_date = pd.to_datetime(df.shipped_date)
    df.order_date_copy = pd.to_datetime(df.order_date_copy)

    # drop 2017 and move data frame year up (2014-2016 to 2015-2017)
    #df = drop2017_and_move2016_up(df)    

    # save the shipped date as index
    df = df.set_index('order_date').sort_index()

    if start2018:
    # data doesn't have enough info about 2017, so we starts from 2018
        df = df.loc['2018':]
    else:
        # keep all but drop 2017 and convert 2014-2016 to 2015-2017
        df =  df
    return df

def drop2017_and_move2016_up(df):
    '''
    This function drops missing year 2017 to combine data with 2018+ dataframe. This is done by creating a temporary
    dataframe and adding a year to years 2014-2016 to creates a seam between 2016 and 2018. 
    return dataframe with new years for temp_df
    '''
    before2017 = df.loc[:'2016'].copy()
    after2017 = df.loc['2018':].copy()
    before2017.index = (before2017.index + pd.Timedelta('1 Y') ).normalize()
    
    return pd.concat([before2017, after2017], axis=0)

def add_date_features(df):
    '''
    Add features based on the date:
    year, month, week number, week day in numerical and human readable values
    
    Parameters:
        df: pandas data frame with date as an index
    Return:
        df: pandas data frame with features added
    '''
    # numerical features
    df['year'] = df.index.year.astype(int)
    df['quarter'] = df.index.quarter.astype(int)
    df['month'] = df.index.month.astype(int)
    df['week'] = df.index.isocalendar().week.astype(int)
    df['day_of_week'] = df.index.day_of_week.astype(int)
    df['day_of_year'] = df.index.day_of_year.astype(int)
    # month and day human readable
    df['month_name'] = df.index.month_name()
    df['day_name'] = df.index.day_name()
    
    return df

def change_customer_type(df):
    '''
    Replace low value count values with 'Other'
    Returns data frame with replaced values in customer type
    '''
    # remove 30 rows with the sales out of Texas
    df = df[df.customer_type != 'Out of State'].copy()
    # make assistance org other
    df.customer_type.replace({'Assistance Org':'Other'},inplace=True)
    
    return df

def change_column_order(df):
    '''
    change the order of columns
    '''
    columns_order = ['customer_name', 
                     'customer_type', 
                     'customer_city',
                     'reseller_name', 
                     'reseller_city',
                     'customer_zip',
                     'order_quantity', 
                     'unit_price',
                     'po_number', 
                     'shipped_date', 
                     'order_date_copy',
                     'month_name', 
                     'day_name',
                     'year',
                     'quarter',
                     'month', 
                     'week',
                     'day_of_week', 
                     'day_of_year',
                     'purchase_amount']
    return df[columns_order]

def get_clean_data(keep2017=False, start2018=start2018_):
    '''
    combines all functions from above
    '''
    df = acquire_data()
    df = basic_clean(df, start2018=start2018)
    df = add_date_features(df)
    df = change_customer_type(df)
    df = change_column_order(df)
    if not keep2017:
        df = drop2017_and_move2016_up(df)

    return df


def split_data(df):
    '''
    splits the data frame based on date
    '''
    train = df.loc[:'2021'].copy()
    test = df.loc['2022'].copy()
    return train, test

def create_customertype_subgroups(train):
    '''
    split data by the customer type
    '''
    types = train[['purchase_amount','customer_type']]
    k_12= types[types["customer_type"]=='K-12']
    local_gov = types[types["customer_type"]=='Local Government']
    state_agency = types[types["customer_type"]=='State Agency']
    higher_ed = types[types["customer_type"]=='Higher Ed']
    other = types[(types['customer_type']=='Assistance Org') | (types['customer_type']=="Other")]
    
    return k_12, local_gov, state_agency, higher_ed, other

def change_XGB_train(X_train_xgb, y_train):
    '''
    removes pandemic data from 2019 and 2020
    '''
    X_before = X_train_xgb.loc[:'2019-06'].copy()
    X_after = X_train_xgb.loc['2020-07':].copy()
    X_before.index = (X_before.index + pd.Timedelta('1 Y') ).normalize()
    y_before = y_train.loc[:'2019-06'].copy()
    y_after = y_train.loc['2020-07':].copy()
    y_before.index = (y_before.index + pd.Timedelta('1 Y') ).normalize()
    XG =  pd.concat([X_before, X_after], axis=0)
    yxg = pd.concat([y_before, y_after], axis=0)
    return XG, yxg

def change_ts(train_ts):
    '''
    removes pandemic data from 2019 and 2020
    Parameters:
        train_ts: train time series
    '''
    before = train_ts.loc[:'2019-06'].copy()
    after = train_ts.loc['2020-07':].copy()
    # one extra day because of the leap year 2020
    before.index = (before.index + pd.Timedelta('1 Y') + pd.Timedelta('1 D')).normalize()

    train_new =  pd.concat([before, after], axis=0)

    #train_new.index = pd.DatetimeIndex(train_new.index).to_period('D')

    return train_new
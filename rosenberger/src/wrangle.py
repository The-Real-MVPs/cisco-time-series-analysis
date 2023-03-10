import pandas as pd
import numpy as np
import os

########### GLOBAL VARIABLES ###########


######## ACQUIRE FUNCTIONS #############

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

def basic_clean(df:pd.DataFrame, start2018=False) -> pd.DataFrame:
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

    # save the shipped date as index
    df = df.set_index('order_date').sort_index()

    if start2018:
    # data doesn't have enough info about 2017, so we starts from 2018
        df = df.loc['2018':]
    else:
        # keep all but drop 2017
        df = pd.concat([df.loc[:'2016'], df.loc['2018':]], axis=0)
    return df

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
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['week'] = df.index.week
    df['day_of_week'] = df.index.day_of_week
    df['day_of_year'] = df.index.day_of_year
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
    df = df[df.customer_type != 'Out of State']
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
                     'year',
                     'quarter',
                     'month_name', 
                     'day_name',
                     'month', 
                     'week',
                     'day_of_week', 
                     'day_of_year',
                     'purchase_amount']
    return df[columns_order]

def get_clean_data(start2018=False):
    '''
    combines all functions from above
    '''
    df = acquire_data()
    df = basic_clean(df, start2018=start2018)
    df = add_date_features(df)
    df = change_customer_type(df)
    df = change_column_order(df)

    return df


def split_data(df, explore=True):
    '''
    splits the data frame based on date
    '''
    if explore:
        train = df.loc[:'2021']
        test = df.loc['2022']
        return train, test
    else:
        return df
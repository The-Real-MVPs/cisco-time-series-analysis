import pandas as pd

'''
The file contains functions that return summury information for:
- whole data frame with purchase_amount sum() for every company in every day
- summury for every customer/reseller/customer_type with purchase_amount min/max/average, date min/max
'''
def get_summary_df(df):
    '''
    Groups by order date, customer name, customer type, customer city and reseller name.
    Calculates the purchase amount by the end. of the day
    
    Parameters:
        df: pandas data frame with the data pulled from DIR site
    Returns:
        pd.DataFrame with the sum of the purchase amount by the end of the day for every company
    '''
    summary_df = pd.DataFrame()
    if 'order_date_copy' in df.columns:
        summary_df = df.groupby(by=\
                        ['order_date_copy', 'customer_name', 'customer_type', 'customer_city', 'reseller_name', 'shipped_date'])\
                        .purchase_amount.sum().to_frame().reset_index()
        summary_df.rename(columns={'order_date_copy':'order_date'}, inplace=True)
        summary_df = summary_df.set_index('order_date').sort_index()
    elif 'order_date' in df.columns:
        summary_df = df.groupby(by=\
                        ['order_date', 'customer_name', 'customer_type', 'customer_city', 'reseller_name', 'shipped_date'])\
                        .purchase_amount.sum().to_frame().reset_index()
        summary_df = summary_df.set_index('order_date').sort_index()  
    return summary_df

def get_summary_orders_df(df):
    '''
    Groups by order date, customer name, customer type, customer city and reseller name.
    Calculates the order quantity of the day
    
    Parameters:
        df: pandas data frame with the data pulled from DIR site
    Returns:
        pd.DataFrame with the sum of the purchase amount by the end of the day for every company
    '''
    summary_df = pd.DataFrame()
    if 'order_date_copy' in df.columns:
        summary_df = df.groupby(by=\
                        ['order_date_copy', 'customer_name', 'customer_type', 'customer_city', 'reseller_name', 'shipped_date'])\
                        .order_quantity.sum().to_frame().reset_index()
        summary_df.rename(columns={'order_date_copy':'order_date'}, inplace=True)
        summary_df = summary_df.set_index('order_date').sort_index()
    elif 'order_date' in df.columns:
        summary_df = df.groupby(by=\
                        ['order_date', 'customer_name', 'customer_type', 'customer_city', 'reseller_name', 'shipped_date'])\
                        .purchase_amount.sum().to_frame().reset_index()
        summary_df = summary_df.set_index('order_date').sort_index()  
    return summary_df

def get_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Creates a pandas data frame with the summary information for every customer:
    - total/min/max/average purchase amount
    - min/max order date
    - min/max shipped date
    
    Parameters:
        df: data frame with data from DIR
    Returns:
        pandas data frame
    '''
    # create a customer summury
    # collect the purchase amount information: total, min, max and average purchase amount
    customer_summary = df.groupby(by='customer_name').purchase_amount.agg(['sum', 'min', 'max', 'mean'])
    # rename columns
    customer_summary.rename(columns={'sum':'total_purchase_amount', 'min':'min_purchase_amount', 
                             'max':'max_purchase amount', 'mean':'average_purchase_amount'},inplace=True)
    # add order date information: first and last order date
    if 'order_date_copy' in df.columns:
        customer_summary = pd.concat([customer_summary,
                           df.groupby(by='customer_name').order_date_copy.agg(['min', 'max'])], axis=1)
    elif 'order_date' in df.columns:
        customer_summary = pd.concat([customer_summary,
                           df.groupby(by='customer_name').order_date.agg(['min', 'max'])], axis=1)        
    # rename columns
    customer_summary.rename(columns={'min':'min_order_date', 
                                 'max':'max_order_date'},inplace=True)
    # add shipping date information: first and last shipping date
    customer_summary = pd.concat([customer_summary,
                           df.groupby(by='customer_name').shipped_date.agg(['min', 'max'])], axis=1)
    # rename columns
    customer_summary.rename(columns={'min':'min_shipping_date', 
                                 'max':'max_shipping_date'},inplace=True)
    return customer_summary

def get_customer_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Creates a pandas data frame with the summary information for every customer type:
    - total/min/max/average purchase amount
    - min/max order date
    - min/max shipped date
    
    Parameters:
        df: data frame with data from DIR
    Returns:
        pandas data frame
    '''
    # create a customer type summury
    # collect the purchase amount information: total, min, max and average purchase amount
    customer_type_summary = df.groupby(by='customer_type').purchase_amount.agg(['sum', 'min', 'max', 'mean'])
    # rename columns
    customer_type_summary.rename(columns={'sum':'total_purchase_amount', 'min':'min_purchase_amount', 
                             'max':'max_purchase amount', 'mean':'average_purchase_amount'},inplace=True)
    # add order date information: first and last order date
    if 'order_date_copy' in df.columns:
        customer_type_summary = pd.concat([customer_type_summary,
                           df.groupby(by='customer_type').order_date_copy.agg(['min', 'max'])], axis=1)
    elif 'order_date' in df.columns:
         customer_type_summary = pd.concat([customer_type_summary,
                           df.groupby(by='customer_type').order_date.agg(['min', 'max'])], axis=1)       
    # rename columns
    customer_type_summary.rename(columns={'min':'min_order_date', 
                                 'max':'max_order_date'},inplace=True)
    # add shipping date information: first and last shipping date
    customer_type_summary = pd.concat([customer_type_summary,
                           df.groupby(by='customer_type').shipped_date.agg(['min', 'max'])], axis=1)
    # rename columns
    customer_type_summary.rename(columns={'min':'min_shipping_date', 
                                 'max':'max_shipping_date'},inplace=True)
    return customer_type_summary

def get_reseller_summary(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Creates a pandas data frame with the summary information for every reseller:
    - total/min/max/average purchase amount
    - min/max order date
    - min/max shipped date
    
    Parameters:
        df: data frame with data from DIR
    Returns:
        pandas data frame
    '''
    # create a reseller summary
    # collect the purchase amount information: total, min, max and average purchase amount
    reseller_summary = df.groupby(by='reseller_name').purchase_amount.agg(['sum', 'min', 'max', 'mean'])
    # rename columns
    reseller_summary.rename(columns={'sum':'total_purchase_amount', 'min':'min_purchase_amount', 
                             'max':'max_purchase amount', 'mean':'average_purchase_amount'},inplace=True)
    # add order date information: first and last order date
    if 'order_date_copy' in df.columns:
        reseller_summary = pd.concat([reseller_summary,
               df.groupby(by='reseller_name').order_date_copy.agg(['min', 'max'])], axis=1)
    elif 'order_date' in df.columns:
        reseller_summary = pd.concat([reseller_summary,
               df.groupby(by='reseller_name').order_date.agg(['min', 'max'])], axis=1)
    # rename columns
    reseller_summary.rename(columns={'min':'min_order_date', 
                                 'max':'max_order_date'},inplace=True)
    # add shipping date information: first and last shipping date
    reseller_summary = pd.concat([reseller_summary,
                                df.groupby(by='reseller_name').shipped_date.agg(['min', 'max'])], axis=1)
    # rename columns
    reseller_summary.rename(columns={'min':'min_shipping_date', 
                                 'max':'max_shipping_date'},inplace=True)
    return reseller_summary
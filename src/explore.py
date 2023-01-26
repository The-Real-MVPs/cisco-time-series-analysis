
# common data science libraries
import numpy as np
import pandas as pd

# vizualizations libraries
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# setting style details for vizualizations
color_pal = sns.color_palette()

plt.style.use('bmh')
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)

# project modules
import src.wrangle as wr



# declaring global variables
df = wr.get_clean_data(start2018=True)
train, test = wr.split_data(df, explore=True)
validate = test.loc[:'2022-06'].copy() 
test = test.loc['2022-07':]

#creating a dataframe for the 'pandemic year'
pandemic_df = df.loc[df.index >= '11-01-2019']
pandemic_df = pandemic_df.loc[pandemic_df.index < '01-01-2021']


def get_df(df):

    df['purchase_amount'] = df['purchase_amount'].astype('int64')
    df['customer_zip'] = df['customer_zip'].astype('int8')
    df['order_quantity'] = df['order_quantity'].astype('int64')
    df['unit_price'] = df['unit_price'].astype('float64')
    
    return df



def q1_viz():
    
    # order for graphs
    months =['January', 'February', 'March', 'April', 
             'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # visualize 
    ax = sns.barplot(data = train, x='day_name', y='purchase_amount', order=days)
    plt.xlabel('Day of the order')
    plt.ylabel('Purchase amount')
    x_left, x_right = ax.get_xlim()
    ax.hlines(train.purchase_amount.mean(), x_left, x_right, ls='--', color='purple')
    plt.title('Average sales for day of the week')
    plt.show()
    
    
def q2_viz():
    
    # visualize 
    ax = sns.barplot(data=train, x='month_name', y='purchase_amount')
    plt.xlabel('Month of the order')
    plt.ylabel('Purchase amount')
    x_left, x_right = ax.get_xlim()
    ax.hlines(train.purchase_amount.mean(), x_left, x_right, ls='--', color='purple')
    plt.title('Average sales for each month')
    plt.show()


def q3_viz():
    
    #sns.set_color_codes("muted")
    ax = sns.barplot(x="quarter", y="purchase_amount", data=train)

    
    
def q4_viz():
    
    # time series variable
    ts = train.purchase_amount
    y_month = ts.resample('M').sum()
    plt.figure(figsize = (20, 6))
    (y_month.diff() / y_month.shift()).plot(alpha=.5, lw=3, c='#1a34ff', 
                                          marker='D', mfc='#f2cb30',mec='black', title='Monthly % Change in Total Sales');
    
    
def q6_vizA():
    
    fix, ax = plt.subplots(figsize = (15,5))
    df['order_quantity'].plot(ax=ax, xlabel='Order Quantity')
    plt.show()
                              
def q6_vizB():
         
    fix, ax = plt.subplots(figsize = (15,5))
    pandemic_df['order_quantity'].plot(ax=ax, xlabel='Order Quantity')
    plt.show()                         
                              
def q6_vizC():
      
    jan2 = pandemic_df.loc[pandemic_df.index == '01-02-2020']
    jan2purchases = jan2.sort_values(by=['order_quantity'], ascending = False).head(6)
    fig, ax = plt.subplots()
    ax.barh(jan2purchases.customer_name, jan2purchases.order_quantity)                          
                              
#def q6_vizD():
                              
#def q6_vizE():

# common data science libraries
import numpy as np
import pandas as pd

# vizualizations libraries
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
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
import src.summaries as s

# declaring global variables
df = s.get_summary_df(wr.get_clean_data(start2018=True))
train, test = wr.split_data(df, explore=True)
validate = test.loc[:'2022-06'].copy() 
test = test.loc['2022-07':]
# create a df with purchase amount by the end of the day only
train_ts = train.purchase_amount.resample('D').sum().to_frame()
# add date features to train_ts
train_ts = wr.add_date_features(train_ts)
# separate by day for stat test
mon = train_ts[train_ts.day_name == 'Monday']
tue = train_ts[train_ts.day_name == 'Tuesday']
wed = train_ts[train_ts.day_name == 'Wednesday']
thu = train_ts[train_ts.day_name == 'Thursday']
fri = train_ts[train_ts.day_name == 'Friday']
sat = train_ts[train_ts.day_name == 'Saturday']
sun = train_ts[train_ts.day_name == 'Sunday']
# create a df with total purchase amount of the month
train_m = train.purchase_amount.resample('M').sum().to_frame()
# add month name
train_m['month_name'] = train_m.index.month_name()

# create a df with total purchase amount of the month
train_q = train.purchase_amount.resample('3M').sum().to_frame()
# add month name
train_q['month_name'] = train_q.index.month_name()
train_q['quarter'] = train_q.index.quarter

# alpha 0.05 for condidence level 95%
alpha = 0.05

#creating a dataframe for the 'pandemic year'
train_pdf,_ = wr.split_data(wr.get_clean_data(start2018=True))
pandemic_df = train_pdf.loc[train_pdf.index >= '11-01-2019']



def get_df(df):

    df['purchase_amount'] = df['purchase_amount'].astype('int64')
    df['customer_zip'] = df['customer_zip'].astype('int8')
    df['order_quantity'] = df['order_quantity'].astype('int64')
    df['unit_price'] = df['unit_price'].astype('float64')
    
    return df

def autopct_format(values):
    '''
    the function accept value_counts from outcome_type
    puts it in % format ready to use in pie charts
    '''
    def my_format(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{:.1f}%)'.format(pct, v=val)
    return my_format

def viz_customer_types():
    '''
    the function creates a pie chart for customer types 
    '''
    piechart_labels = ['Local Governments','School Districts', 'Higher Education Institutions', 'State Agencies', 'Others']
    values = train.customer_type.value_counts().tolist()
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=piechart_labels,
            colors=sns.color_palette('Set2'), autopct=autopct_format(values),
            shadow=False)
    plt.title('Customer types')
    plt.show()

def q1_viz_per_order():
    
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
    plt.title('Average purchase amount per day')
    plt.show()

def q1_viz():
    
    # order for graphs
    months =['January', 'February', 'March', 'April', 
             'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # visualize 
    ax = sns.barplot(data = train_ts, x='day_name', y='purchase_amount', order=days)
    plt.xlabel('Day of the order')
    plt.ylabel('Purchase amount')
    x_left, x_right = ax.get_xlim()
    ax.hlines(train_ts.purchase_amount.mean(), x_left, x_right, ls='--', color='purple')
    plt.title('Average sales for day of the week')
    plt.show()

def q1_ttest():
    '''
    One sample T-test. Calculates t and p values for every day of the week and saves them into a data frame.
    Returns data frame
    '''
    days_of_week = {'Monday':mon, 'Tuesday':tue, 'Wednesday':wed, 'Thursday':thu, 
                    'Friday':fri, 'Saturday':sat, 'Sunday':sun}
    ttest_results_days = pd.DataFrame(columns=['Day', 'T-value', 'P-value'])
    for day in days_of_week:
        sale = days_of_week[day]['purchase_amount']
        t, p = stats.ttest_1samp(sale, train_ts.purchase_amount.mean())
        ttest_results_days.loc[len(ttest_results_days)] = [day, t, p]
    ttest_results_days.set_index('Day', inplace=True)
    return ttest_results_days

def q1_anova():
    '''
    run Anova test to compare means of sales on Mon, Tue and Wed
    '''
    # define p-value
    p = 1
    
    # check for equal varicances. Run kruskall wallis or anova tests
    if stats.levene(mon.purchase_amount, tue.purchase_amount, wed.purchase_amount)[1] < alpha:
        print('Variances are different. Use Kruskall Wallis')
        _, p = stats.kruskal(mon.purchase_amount, tue.purchase_amount, wed.purchase_amount)
    else:
        print('Variances are equal. Use ANOVA')
        _, p = stats.f_oneway(mon.purchase_amount, tue.purchase_amount, wed.purchase_amount)
    print('====================')
    if p < alpha:
        print('Reject null hypothesis')
        print('There is a significant difference in means of sales during work days Monday through Wednesday')
    else:
        print('Fail to reject null hypothesis')
        print('There is no significant difference in means of sales during work days Monday through Wednesday')
    
def q2_viz():
    
    # visualize 
    ax = sns.barplot(data=train_m, x='month_name', y='purchase_amount')
    plt.xlabel('Month of the order')
    plt.ylabel('Purchase amount')
    x_left, x_right = ax.get_xlim()
    ax.hlines(train_m.purchase_amount.mean(), x_left, x_right, ls='--', color='purple')
    plt.title('Average sales for each month')
    plt.show()

def q2_ttest():
    '''
    Run one sample T-test to check in which month sales are significantly different from the average monthly sales
    '''
    months =['January', 'February', 'March', 'April', 'May', 
             'June', 'July', 'August', 'September', 'October', 'November', 'December']
    ttest_results_months = pd.DataFrame(columns=['Month', 'T-value', 'P-value'])
    for m in months:
        month = train_m[train_m.month_name == m]['purchase_amount']
        t, p = stats.ttest_1samp(month, train_m.purchase_amount.mean())
        ttest_results_months.loc[len(ttest_results_months)] = [m, t, p]
    ttest_results_months.set_index('Month', inplace=True)
    return ttest_results_months.sort_values(by='T-value', ascending=False)

def q3_viz():
    
    #sns.set_color_codes("muted")
    ax = sns.barplot(x="quarter", y="purchase_amount", data=train_q)
    x_left, x_right = ax.get_xlim()
    ax.hlines(train_q.purchase_amount.mean(), x_left, x_right, ls='--', color='purple')

def q3_ttest():
    '''
    run stat test for quarters
    '''
    ttest_results_quarters = pd.DataFrame(columns=['Quarter', 'T-value', 'P-value'])
    for i in range(1, 5):
        q = train_q[train_q.quarter == i]['purchase_amount']
        t, p = stats.ttest_1samp(q, train_q.purchase_amount.mean())
        ttest_results_quarters.loc[len(ttest_results_quarters)] = [i, t, p]
    ttest_results_quarters.set_index('Quarter', inplace=True)
    return ttest_results_quarters   
    
def q4_viz():
    '''
    plot monthly % change in sales
    '''
    # time series variable
    ts = train.purchase_amount
    y_month = ts.resample('M').sum()
    plt.figure(figsize = (20, 6))
    (y_month.diff() / y_month.shift()).plot(alpha=.5, lw=3, c='#1a34ff', 
                                          marker='D', mfc='#f2cb30',mec='black', title='Monthly % Change in Total Sales');
    
    
def q5_vizA():
    '''
    shows viz with huge order quantity spike.
    '''
    fix, ax = plt.subplots(figsize = (15,5))
    train_pdf['order_quantity'].plot(ax=ax, xlabel='Order Quantity')
    plt.title('Order quantity outlier')
    plt.show()

def q5_thhsc():
    '''
    closer look at the outlier
    '''
    orders_summary = s.get_summary_orders_df(train_pdf)
    sales_summary = s.get_summary_df(train_pdf)
    thhsc = pd.concat([
        orders_summary[orders_summary.customer_name == 'Texas Health and Human Services Commission'],
        sales_summary[sales_summary.customer_name == 'Texas Health and Human Services Commission'].purchase_amount
    ],axis=1)
    #ax, _ = plt.subplots(figsize=(18,6))
    ax = thhsc.order_quantity.plot(label='Number of orders', lw=2, c='#1a34ff', alpha=0.6)
    ax = thhsc.purchase_amount.plot(label='Purchase amount', lw=1.5, c='#b56b35', alpha=0.7)
    plt.title('Texas Health and Human Services Commission orders')
    ax.set(yticks=[0, 1_000_000, 2_000_000, 3_000_000, 4_000_000])
    ax.set(yticklabels=['0', '1M', '2M', '3M', '4M'])
    ax.set(xlabel=None)
    plt.legend()
    plt.show()
                              
def q5_vizB():
         
    fix, ax = plt.subplots(figsize = (15,5))
    pandemic_df['order_quantity'].plot(ax=ax, xlabel='Order Quantity')
    plt.show()                         
                              
def q5_vizC():
      
    jan2 = pandemic_df.loc[pandemic_df.index == '01-02-2020']
    jan2purchases = jan2.sort_values(by=['order_quantity'], ascending = False).head(6)
    fig, ax = plt.subplots()
    ax.barh(jan2purchases.customer_name, jan2purchases.order_quantity)                          
                              
#def q6_vizD():
                              
#def q6_vizE():
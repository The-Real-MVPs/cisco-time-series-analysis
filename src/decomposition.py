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

plt.style.use("seaborn-whitegrid")
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

start2018_ = wr.get_start2018()
# declaring global variables
df = s.get_summary_df(wr.get_clean_data(start2018=False))
df = wr.drop2017_and_move2016_up(df)
train, _ = wr.split_data(df)

# show df from 2018 for negative higher education trend
df2018 = s.get_summary_df(wr.get_clean_data(start2018=True))
train2018, _ = wr.split_data(df2018)

y_daily = train.purchase_amount.resample('D').sum()
y_weekly = train.purchase_amount.resample('W').sum()
y_monthly = train.purchase_amount.resample('M').sum()

k_12, local_gov, state_agency, higher_ed, other = wr.create_customertype_subgroups(train)

_, _, _, higher_ed2018, _ = wr.create_customertype_subgroups(train2018)

# daily decompostions for all types
result_d = sm.tsa.seasonal_decompose(y_daily)
decomposition_d = pd.DataFrame({
    'y': result_d.observed,
    'trend': result_d.trend,
    'seasonal': result_d.seasonal,
    'resid': result_d.resid,
})

# weekly decompostions for all types
result_w = sm.tsa.seasonal_decompose(y_weekly)
decomposition_w = pd.DataFrame({
    'y': result_w.observed,
    'trend': result_w.trend,
    'seasonal': result_w.seasonal,
    'resid': result_w.resid,
})

# mothly decomposition for all types
result_m = sm.tsa.seasonal_decompose(y_monthly)
decomposition_m = pd.DataFrame({
    'y': result_m.observed,
    'trend': result_m.trend,
    'seasonal': result_m.seasonal,
    'resid': result_m.resid,
})
decomposition_m['time_dummy'] = np.arange(len(decomposition_m.index))

# monthly decomposition schools
k_12m = k_12.purchase_amount.resample('M').sum()
result_k_12m = sm.tsa.seasonal_decompose(k_12m)
decomposition_k_12m = pd.DataFrame({
    'y': result_k_12m.observed,
    'trend': result_k_12m.trend,
    'seasonal': result_k_12m.seasonal,
    'resid': result_k_12m.resid,
})
# monthly decomposition schools before 2018
k_12_before = k_12.loc[:'2020']
k_12m_before = k_12_before.purchase_amount.resample('M').sum()
result_k_12m_before = sm.tsa.seasonal_decompose(k_12m_before)
decomposition_k_12m_before = pd.DataFrame({
    'y': result_k_12m_before.observed,
    'trend': result_k_12m_before.trend,
    'seasonal': result_k_12m_before.seasonal,
    'resid': result_k_12m_before.resid,
})

# monthly decomposition higher education
higher_ed_m = higher_ed.purchase_amount.resample('M').sum()
result_higher_ed_m = sm.tsa.seasonal_decompose(higher_ed_m)
decomposition_higher_ed_m = pd.DataFrame({
    'y': result_higher_ed_m.observed,
    'trend': result_higher_ed_m.trend,
    'seasonal': result_higher_ed_m.seasonal,
    'resid': result_higher_ed_m.resid,
})

# monthly decomposition higher education since 2018
higher_ed_m2018 = higher_ed2018.purchase_amount.resample('M').sum()
result_higher_ed_m2018 = sm.tsa.seasonal_decompose(higher_ed_m2018)
decomposition_higher_ed_m2018 = pd.DataFrame({
    'y': result_higher_ed_m2018.observed,
    'trend': result_higher_ed_m2018.trend,
    'seasonal': result_higher_ed_m2018.seasonal,
    'resid': result_higher_ed_m2018.resid,
})

# monthly decomposition local governments
local_gov_m = local_gov.purchase_amount.resample('M').sum()
result_local_gov_m = sm.tsa.seasonal_decompose(local_gov_m)
decomposition_local_gov_m = pd.DataFrame({
    'y': result_local_gov_m.observed,
    'trend': result_local_gov_m.trend,
    'seasonal': result_local_gov_m.seasonal,
    'resid': result_local_gov_m.resid,
})

# monthly decomposition state agencies
state_agency_m = state_agency.purchase_amount.resample('M').sum()
result_state_agency_m = sm.tsa.seasonal_decompose(local_gov_m)
decomposition_state_agency_m = pd.DataFrame({
    'y': result_state_agency_m.observed,
    'trend': result_state_agency_m.trend,
    'seasonal': result_state_agency_m.seasonal,
    'resid': result_state_agency_m.resid,
})



##### q 2 trend line
def viz_monthly_trend():
    '''
    Creates seasonal decomposition and plots the trend line of total monthly purchase amount
    '''

    ax = sns.regplot(y=decomposition_m.trend, x=decomposition_m.time_dummy)
    plt.title('Monthly purchase amount trend')
    plt.xlabel(None)
    ax.set(xticks=[7, 19, 31, 43, 55, 67])
    ax.set(xticklabels=['Jan 2015', 'Jan 2016', 'Jan 2018', 'Jan 2019', 'Jan 2020', 'Jan 2021'])
    plt.show()

def show_all_trends():
    '''
    show monthly trends for all customer types except 'others'
    '''
    k12_trend = decomposition_k_12m.trend
    hedu_trend = decomposition_higher_ed_m.trend
    loc_gov_trend = decomposition_local_gov_m.trend
    state_agency_trend = decomposition_state_agency_m.trend
    
    overall_trend = decomposition_m.trend
    # create time dummies
    time_dummy = decomposition_m.time_dummy
    ax = sns.regplot(x=time_dummy, y=k12_trend, scatter=False, label='School districts')
    ax = sns.regplot(x=time_dummy, y=hedu_trend, scatter=False, label='Higher Education')
    ax = sns.regplot(x=time_dummy, y=loc_gov_trend, scatter=False, label='Local goverments', line_kws={'alpha':0.3})
    ax = sns.regplot(x=time_dummy, y=state_agency_trend, scatter=False, label='State Agencies', line_kws={'alpha':0.3})


    ax.set(xticks=[7, 19, 31, 43, 55, 67])
    ax.set(xticklabels=['Jan 2015', 'Jan 2016', 'Jan 2018', 'Jan 2019', 'Jan 2020', 'Jan 2021'])
    plt.xlabel(None)
    ax.set(yticks=[4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000])
    ax.set(yticklabels=['$4M', '$5M', '$6M', '$7M', '$8M'])
    plt.ylabel('Monthly purchases')
    plt.legend()
    plt.title('Monthly purchase amount trends')
    plt.show()

def show_hedu_trend2018():
    '''
    show trend for 2018 for higher education subtype
    '''
    decomposition_higher_ed_m2018['time_dummy'] = np.arange(len(decomposition_higher_ed_m2018.index))
    ax = sns.regplot(data=decomposition_higher_ed_m2018, x='time_dummy', y='trend')
    plt.title('Monthly purchases trend for Higher Education Institutions after 2018')
    plt.xlabel(None)
    ax.set(xticks=[13, 23, 33])
    ax.set(xticklabels=['Jan 2019', 'Jan 2020', 'Jan 2021'])
    ax.set(yticks=[])
    plt.show()

def show_school_trend_before():
    '''
    show the school's trend before 2020
    '''
    decomposition_k_12m_before['time_dummy'] = np.arange(len(decomposition_k_12m_before.index))
    ax = sns.regplot(data=decomposition_k_12m_before, x='time_dummy', y='trend')
    plt.title('Monthly purchases trend for School Districts before COVID-2019')
    plt.xlabel(None)
    ax.set(xticks=[7, 19, 31, 43, 55])
    ax.set(xticklabels=['Jan 2015', 'Jan 2016', 'Jan 2018', 'Jan 2019', 'Jan 2020'])
    ax.set(yticks=[])
    plt.show()

def viz_school_purchases():
    '''
    Creates a plot with 
    '''
    ax = k_12m.loc[:'2020'].plot(label='Before COVID-2019', lw=2)
    ax = k_12m.loc['2020':].plot(label='After COVID-2019', lw=2)
    plt.title('Monthly total purchase amount for school districts before and after COVID-2019')
    ax.set(yticks=[10_000_000, 20_000_000])
    ax.set(yticklabels=['$10M', '$20M'])
    plt.legend()
    plt.show()
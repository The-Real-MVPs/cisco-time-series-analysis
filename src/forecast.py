import pandas as pd
import numpy as np

# models
import xgboost as xgb

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# import helper modules
import src.wrangle as wr
import src.summaries as s

# import graphic modules
import matplotlib.pyplot as plt
import seaborn as sns
# set default parameters
plt.style.use("seaborn-whitegrid") # returns warnings
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
#plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
plt.rcParams.update({'figure.dpi':120})
#%config InlineBackend.figure_format = 'retina'

# set default parameters for floats
pd.options.display.float_format = '{:,.2f}'.format

# generate indexes for every day in 2023, create a data frame
forecast_df = pd.DataFrame(index=pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'))
# add date features
forecast_df = wr.add_date_features(forecast_df)

# features to keep in data for training model
features = ['month', 'week', 'day_of_week', 'year','day_of_year']

# load data
df = s.get_summary_df(wr.get_clean_data())
df = wr.drop2017_and_move2016_up(df)
# resample by day
train = df.purchase_amount.copy().resample('D').sum().to_frame()
# train ONLY after June 2020, to reduce overestimating
train = train.loc['2020-06':]
# add date features for the train set
train = wr.add_date_features(train)
# create train, validate, test sets
X_train = train[features]
y_train = train.purchase_amount

X_validate = X_train.loc['2022'].copy()
X_train = X_train.loc[:'2021']
y_validate = y_train.loc['2022'].copy()
y_train = y_train.loc[:'2021'].copy()

X_test = forecast_df[features]

# create a model
xgb_model = xgb.XGBRegressor(n_estimators = 500, 
                         early_stopping_rounds = 25,
                         learning_rate=0.01, verbosity=0)
xgb_model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_validate, y_validate)], verbose=False)

# create a forecast
forecast = pd.Series(xgb_model.predict(X_test), index=X_test.index)

def plot_2023_forecast():
    '''
    plots the forecast of total purchase amount for 2023
    '''
    plt.figure(figsize=(11,4))
    ax = forecast.plot(alpha=0.7)
    ax.set(yticks=[0, 200_000, 400_000, 600_000, 800_000, 1_000_000, 1_200_000])
    ax.set(yticklabels=['0', '200K', '400K', '600K', '800K', '1M', '1.2M'])
    plt.title('Our forecast for 2023')
    plt.text(x='2023-01-10', y = 900_000, s=f'Total purchase amount for 2023 is ${(forecast.sum()):,.00f}')
    plt.show()

def plot_all_df_and_forecast():
    '''
    plot all data and forecast
    '''
    plt.figure(figsize=(11, 4))
    ax = df.purchase_amount.resample('M').sum().plot(alpha=0.7, label='Data')
    ax = forecast.resample('M').sum().plot(label='Forecast')
    ax.set(yticks=[0, 10_000_000, 20_000_000, 30_000_000])
    ax.set(yticklabels=['0', '10M', '20M', '30M'])
    ax.set(xticks=['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023'])
    ax.set(xticklabels=['2015', '2016', '2018', '2019', '2020', '2021', '2022', '2023'])
    ax.set(ylabel='Total monthly purchase amount')
    plt.title('Our forecast for 2023')
    plt.legend()
    plt.show()
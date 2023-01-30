import pandas as pd
import numpy as np

# models
import statsmodels.api as sm
# pmdarima and prophet need installation!
#import pmdarima as pm # pip install pmdarima
import prophet # python -m pip install prophet
from prophet import Prophet
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
#from pmdarima.arima.utils import ndiffs

from sklearn.metrics import mean_squared_error

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

##### get the data
df = s.get_summary_df(wr.get_clean_data(start2018=True))
# define target vaiable
target = 'purchase_amount'
#### split into train, validate, test sets
train, test = wr.split_data(df)
validate = test.loc[:'2022-06'].copy() 
test = test.loc['2022-07':].copy()
# get time series with the daily resample
X_train_ts = train.purchase_amount.copy().resample('D').sum()
X_validate_ts = validate.purchase_amount.copy().resample('D').sum()
X_test_ts = test.purchase_amount.copy().resample('D').sum()
# data frames out of the TS
X_train = X_train_ts.to_frame()
X_validate = X_validate_ts.to_frame()
X_test = X_test_ts.to_frame()
# data frames to use with Prophet
pr_train = X_train.reset_index()
pr_train.columns = ['ds', 'y']
pr_validate = X_validate.reset_index()
pr_validate.columns = ['ds', 'y']
# data frames to keep predictions
predictions_train = X_train[target].to_frame()
predictions_validate = X_validate[target].to_frame()
# data frame to keep scores
scores = pd.DataFrame(columns=['model_name', 'train_score', 'validate_score'])

# baseline = 654_835.73
baseline = X_train_ts.mean()
# add baseline values to predictions
predictions_train['baseline'] = baseline
predictions_validate['baseline'] = baseline

def show_ts():
    '''
    plots daily sales for the X_train
    '''
    ax = X_train_ts.plot()
    plt.title('Daily sales')
    ax.set(yticks=[0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_000_000])
    ax.set(yticklabels=['0', '1M', '2M', '3M', '4M', '5M', '6M', '7M'])
    plt.show()

def evaluate(target_name: str = target, model_name: str = 'baseline'):
    '''
    Calculate RMSE score for train and validate predictions,
    store RMSE into scores data frame,
    plots the predictions vs actual values

    Parameters:
        target_name: str, target variable
        model_name: str, model name, should match the column name 
                    in prediction data frame where values are stored
    Returns:
        no returns
    '''
    RMSE_train = round(np.sqrt(mean_squared_error(X_train[target_name], predictions_train[model_name])))
    RMSE_validate = round(np.sqrt(mean_squared_error(X_validate[target_name], predictions_validate[model_name])))
    scores.loc[len(scores)] = [model_name, RMSE_train, RMSE_validate]
    
    # plot
    plt.figure(figsize = (12,4))
    plt.plot(X_train[target_name], label='Train', linewidth=1)
    plt.plot(X_validate[target_name], label='Validate', linewidth=1)
    plt.plot(predictions_train[model_name], label=model_name + '_train')
    plt.plot(predictions_validate[model_name], label=model_name + '_validate')
    plt.legend()
    plt.title(target_name)
    
    print(target_name, '-- RMSE train: {:.0f}'.format(RMSE_train))
    print(target_name, '-- RMSE validate: {:.0f}'.format(RMSE_validate))
    plt.show()

def evaluate_rmse(target_name:str, model_name: str, train=True):
    '''
    Calculate RMSE score for train and validate predictions,
    store RMSE into scores data frame,

    Parameters:
        target_name: str, target variable
        model_name: str, model name, should match the column name 
                    in prediction data frame where values are stored
        train: bool, True if train set has to be evaluated
    Returns:
        no returns
    '''
    if train:
        RMSE_train = round(np.sqrt(mean_squared_error(predictions_train[target_name], predictions_train[model_name])))
        RMSE_validate = round(np.sqrt(mean_squared_error(predictions_validate[target_name], predictions_validate[model_name])))
        scores.loc[len(scores)] = [model_name, RMSE_train, RMSE_validate]
    else:
        RMSE_validate = round(np.sqrt(mean_squared_error(predictions_validate[target_name], predictions_validate[model_name])))
        scores.loc[len(scores)] = [model_name, None, RMSE_validate]
    
def plot_model(target_name: str, model_name: str):
    '''
    Plots the predictions vs actual values
    Parameters:
        target_name: str, target variable
        model_name: str, model name, should match the column name 
                    in prediction data frame where values are stored
    Returns:
        no returns
    '''
    plt.figure(figsize = (12,4))
    plt.plot(X_train[target_name], label='Train', linewidth=1)
    plt.plot(X_validate[target_name], label='Validate', linewidth=1)
    plt.plot(predictions_train[model_name], label=model_name + '_train')
    plt.plot(predictions_validate[model_name], label=model_name + '_validate')
    plt.title(target_name)
    
    print(target_name, '-- RMSE train: {:.0f}'.format(RMSE_train))
    print(target_name, '-- RMSE validate: {:.0f}'.format(RMSE_validate))
    plt.show()

def moving_average(span=2):
    '''
    Create moving averages.
    Saves results to predictions_train and predictions_validate
    Evaluate results and saves scores into scores data frame
    
    Parameters:
        span: moving average period
    '''
    # assign the period for the moving average
    span = span
    # identify the model
    model_name = 'Moving Average' + ' ' + str(span)
    # create a baseline value of moving average
    ma_baseline = round(X_train.purchase_amount.rolling(span).mean()[-1], 2)
    
    # rolling amounts to fill train and validate sets
    rolling_amount_train = round(X_train.purchase_amount.rolling(span).mean(), 2).fillna(ma_baseline)
    rolling_amount_validate = round(X_validate.purchase_amount.rolling(span).mean(), 2)
    # slicing index to replace the indexes in the beginning of validation set with the last values of train set
    temp_index = span - 1
    # put rolling moving average values to the predictions train
    predictions_train[model_name] = rolling_amount_train
    # replace validate nulls with last values of train set
    rolling_amount_validate[:temp_index] = predictions_train[model_name][-temp_index:].values
    # put predictions into predictions validate
    predictions_validate[model_name] = rolling_amount_validate
    # evaluate and save results into scores data frame
    evaluate_rmse(target, model_name)

def exponential_moving_average(span=3):
    '''
    Create exponential moving averages.
    Saves results to predictions_train and predictions_validate
    Evaluate results and saves scores into scores data frame
    
    Parameters:
        span: moving average period
    '''
    # identify the model
    model_name = 'Exp Moving Average' + ' ' + str(span)
    predictions_train[model_name] = X_train.ewm(span=span).mean()
    predictions_validate[model_name] = X_validate.ewm(span=span).mean()
    evaluate_rmse(target, model_name)

def show_ma_baseline_scores(min_limit:int = 2, max_limit:int = 2):
    '''
    Calulates MA (moving average) and EMA (exponential moving average).
    Display scores data frame
    Parameters:
        min_limit: int, minimum value of spans(periods) for MA and EMA
        max_value: int, minimum value of spans(periods) for MA and EMA
    Returns:
        void
        * We already know that the best scores are for the span=2, 
        so we set deafault limits as 2.
    '''
    evaluate_rmse(target, 'baseline')
    for i in range(min_limit, max_limit+1):
        moving_average(i)
        exponential_moving_average(i)
        display(scores)

#### ARIMA MODEL
def model_arima(p:int=1, d:int=0, q:int=0, show_viz:bool=False):
    '''
    Create ARIMA model, fit results to train time series, predict validate time series
    Evaluate results
    Plot the predictions for the validate set agains the actual value by request
    Parameters are parameters for ARIMA model:
        p: int, the number of lag observations in the model, also known as the lag order. 
        d: int, the number of times the raw observations are differenced; also known as the degree of differencing. 
        q: int, the size of the moving average window, also known as the order of the moving average.
        show_viz: bool. if True - plots predictions
    * default parameters for daily predictions are set to 1, 0, 0
    1 - the only lag that has correlation is 1 day
    0 - data is stationary, no trend
    0 - no MA window
    '''
    # create ARIMA model
    model = ARIMA(X_train_ts, order=(p, d, q))
    # model name to save predictions 
    model_name = f'ARIMA {p},{d},{q}'
    fit = model.fit()
    forecast = fit.forecast(len(X_validate_ts))
    predictions_validate[model_name] = forecast
    evaluate_rmse(target, model_name, train=False)
    if show_viz:
        X_validate_ts.plot(alpha = 0.3)
        forecast.plot(label='predictions', alpha=0.3)
        predictions_validate.baseline.plot(label='Baseline', ls='--')
        predictions_validate['Exp Moving Average 2'].plot(label='Exponential MA')
        plt.title(f'Predictions of {model_name}')
        plt.legend()
        plt.show()

def create_arima_models():
    model_arima()
    model_arima(2, 0, 0)
    model_arima(0, 0, 2)
    model_arima(0, 0, 5)
    display(scores)

#### PROPHET

def model_prophet(linear=False, linear_ranges=[0.2, 0.3, 0.5, 0.6, 0.7, 0.8]):
    '''
    
    '''
    period = len(pr_validate)
    
    if linear:
        for r in linear_ranges:
            # create the model
            # linear ranges to put into changepoint_range, identify proportion of historical data
            # interval_width for the confidence interval, didn't change any values though
            model = Prophet(growth='linear', changepoint_range=r, interval_width=0.95)
            # fit on train
            model.fit(pr_train)
            # make forecast
            # periods = len of validation of test sets. 181 for daily forecasts, 27 for weekly
            future = model.make_future_dataframe(periods=period, freq='D')
            forecast = model.predict(future)
            # save forecast to prediction data frame and evaluate them
            model_name = 'Linear Prophet ' + str(r)
            predictions_train[model_name] = forecast.iloc[0:len(pr_train)].yhat.tolist()
            predictions_validate[model_name] = forecast.iloc[len(pr_train):].yhat.tolist()
            evaluate_rmse(target, model_name)
    else:
        # if not linear, use 'flat' growth, changepoint_range doesn't matter in this case, use default 0.8
        model = Prophet(growth='flat', interval_width=0.95)
        # fit on train
        model.fit(pr_train)
        # make forecast
        # periods = len of validation of test sets. 181 for daily forecasts, 27 for weekly
        future = model.make_future_dataframe(periods=period, freq='D')
        forecast = model.predict(future)
        # save forecast to prediction data frame and evaluate them
        model_name = 'Prophet'
        predictions_train[model_name] = forecast.iloc[0:len(pr_train)].yhat.tolist()
        predictions_validate[model_name] = forecast.iloc[len(pr_train):].yhat.tolist()
        evaluate_rmse(target, model_name)

def create_prophet():
    model_prophet()
    model_prophet(linear=True)

###### run Prophet on test set

def get_prophet_data(df:pd.DataFrame):
    '''
    Creates train and test data sets for the final model implementation 
    Parameters:
        df: pd.DataFrame, clean original or summary data frame before the split
    '''
    # make different split. train now includes data from train and validation sets
    #train = df.loc[:'2022-06'].copy() 
    #test = df.loc['2022-07':].copy()

    # prepare data frames for the Prophet model
    pr_train = validate.purchase_amount.copy().resample('D').sum()\
                        .to_frame()\
                        .reset_index()\
                        .rename({'order_date':'ds', 'purchase_amount':'y'}, axis=1)
    pr_test = test.purchase_amount.copy().resample('D').sum()\
                        .to_frame()\
                        .reset_index()\
                        .rename({'order_date':'ds', 'purchase_amount':'y'}, axis=1)
    return pr_train, pr_test

def run_test_model(df):
    '''
    Calls get_prophet_data to get train and validate sets
    Creates prophet model
    Saves forecasts to train and test sets

    Parameters:
        df: cleaned original or summary data frame
    Returns:
        pr_train: pd.DataFrame with original data, baseline and predictions
        pr_test: pd.DataFrame with original data, baseline and predictions
    '''
    pr_train, pr_test = get_prophet_data(df)
    # set the length of forecast
    period = len(pr_test)
    model = Prophet(growth='flat', interval_width=0.95)
    # fit on train
    model.fit(pr_train)
    # make forecast
    # periods = len of validation of test sets. 181 for daily forecasts, 27 for weekly
    future = model.make_future_dataframe(periods=period, freq='D')
    forecast = model.predict(future)
    # save forecast to prediction data frame and evaluate them
    model_name = 'Prophet'
    forecast_train = forecast.iloc[0:len(pr_train)].yhat.tolist()
    forecast_test = forecast.iloc[len(pr_train):].yhat.tolist()
    pr_train['forecast'] = forecast_train
    pr_test['forecast'] = forecast_test
    pr_test['baseline'] = baseline
    
    return pr_train, pr_test

def get_test_scores(pr_test:pd.DataFrame):
    '''
    Calculates RMSE scores for the test data
    Parameters:
        pr_test: pd.DataFrame, test data set with forecast and baseline columns
    '''
    
    RMSE_prohet = np.sqrt(mean_squared_error(pr_test.y, pr_test.forecast))
    RMSE_baseline = np.sqrt(mean_squared_error(pr_test.y, pr_test.baseline))
    # set model name as an index in scores data frame
    scores.set_index('model_name', inplace=True)
    test_scores = pd.DataFrame({
                         'Baseline':[ scores.loc['baseline', 'train_score'], 
                                     scores.loc['baseline', 'validate_score'], RMSE_baseline],
                         'Prophet':[ scores.loc['Prophet', 'train_score'],
                                    scores.loc['Prophet', 'validate_score'], RMSE_prohet]
                            },index=[ 'Train RMSE', 'Validate RMSE', 'Test RMSE'])
    return test_scores

def plot_predictions(pr_test):
    '''
    Plot Prophet Predictions
    '''
    sns.set_style("whitegrid")
    ax = sns.lineplot(data=pr_test, x='ds',y='y', label='Actual values')
    ax = sns.lineplot(data=pr_test, x='ds',y='forecast', label='Model forecast')
    ax = sns.lineplot(data=pr_test, x='ds',y='baseline', label='Baseline predictions')
    ax.set(title='Model predictions on the test set')
    ax.set(yticks=[0, 500_000, 2_000_000, 3_500_000])
    ax.set(yticklabels=['0', '500K', '2M', '3.5M'])
    ax.set(ylabel='purchase_amount')
    plt.show()

def show_test():
    '''
    Runs evaluation and plot functions for the test data
    '''
    _, pr_test = run_test_model(df)
    test_scores = get_test_scores(pr_test)
    display(test_scores)
    plot_predictions(pr_test)
    



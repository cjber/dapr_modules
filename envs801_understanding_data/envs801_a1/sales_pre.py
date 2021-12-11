# %%
'''
# Data Preprocessing

First the libraries used for the preprocessing were read in. This section first cleans the store data, then moves onto cleaning both the testing and training data together.
'''

# %%
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # visualisations
import seaborn as sns  # more vis
import pandas as pd  # for working with dataframes
import statsmodels.api as sm  # stats functions

from datetime import date, datetime  # working with datetime types
from typing import Dict  # type checking
# --

# %%
'''
## Store data

First `store.csv` was read in, containing information regarding the stores in this analysis. The final two columns in `store.csv` are empty and unlabelled so are removed. Store provides the year and month the nearest competition opened with the `CompetitionOpenSinceYear` and `CompetitionOpenSinceMonth` variables, making the assumption that all stores opened on the first day of the month, these can be converted into a single variable with the `datetime` type called `CompetitionDate`. This will make working with the data as a timeseries easier.

Similarly the `Promo2Start` variable may be created from the `Promo2SinceYear` and `Promo2SinceWeek` variables, using a week is decidedly more complex, particularly as the number of weeks in a year is 52.1775, meaning the number of weeks in a year are either considered to be 52, or 53. The [International Standards Organisation](https://www.iso.org) provides the ISO week date system as part of the ISO 8601 datetime standard, which since `pandas` 3.8 is accessible through the `fromisocalendar` method. This method may be applied to every non `NULL` row in the dataframe to convert the week and year provided into a `datetime` variable type, with the assumption that the promotion started on the first day of the week (Monday). As `Promo2` runs for 3 months, indicated by the data documentation, and repeats every 3 months (`PromoInterval`), it can be assumed that once, `Promo2` has been initiated, it will always be running.
'''

# %%
# read in store data, final two columns are untitled and blank
store = pd.read_csv("~/data/modules/envs801/DA1920_store.csv",
                    low_memory=False).iloc[:, :-2]


# create competition start date as datetime
store['CompetitionDate'] = pd.to_datetime(
    {'year': store['CompetitionOpenSinceYear'],
     'month': store['CompetitionOpenSinceMonth'],
     'day': 1}
)


def week_year_to_datetime(row: pd.Series) -> pd.Series:
    return np.datetime64(
        date.fromisocalendar(int(row['Promo2SinceYear']),
                             int(row['Promo2SinceWeek']), 1)
    )


store['Promo2Start'] = store.apply(
    lambda row: week_year_to_datetime(row)
    if row.notnull().all() else None, axis=1
)

store = pd.get_dummies(store, columns=['Assortment'], drop_first=True)
# --

# %%
'''
Competition distance was highly skewed (Figure \ref{fig:hist}), so the log of the distance was taken, which provided a better distribution.
'''

# %%
store['log_competition_dist'] = np.log1p(store['CompetitionDistance'])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
store['CompetitionDistance'].hist(bins=20, edgecolor='k', ax=axes[0])
axes[0].axvline(store['CompetitionDistance'].quantile(),
                color='k', linestyle='dashed')
store['log_competition_dist'].hist(bins=20, edgecolor='k', ax=axes[1])
axes[1].axvline(store['log_competition_dist'].quantile(),
                color='k', linestyle='dashed')
plt.show(block=False)
# --

# %%
'''
Now all unused variables could be dropped.
'''

# %%
store.drop(['PromoInterval',
            'CompetitionOpenSinceMonth',
            'CompetitionOpenSinceYear',
            'Promo2SinceWeek',
            'Promo2SinceYear',
            'CompetitionDistance'], axis=1, inplace=True)
# --

# %%
'''
## Test and training data

Following the store data cleaning, the training and testing data was read in. When reading in the CSVs, the `Date` columns were parsed as the `datetime` type. These dataframes could be easily matched based on the `Store` variable which contains unique numerical IDs for each store in the data.
'''

# %%
# merge train and store using 'Store'
train = pd.read_csv("~/data/modules/envs801/DA1920_train.csv",
                    low_memory=False,
                    parse_dates=['Date']
                    )
train = train[(train['Date'] >= '2013-01-01') &
              (train['Date'] <= '2015-07-31')]
train = train.merge(store, on='Store')

test = pd.read_csv("~/data/modules/envs801/DA1920_test.csv",
                   low_memory=False,
                   parse_dates=['Date'])
test = test[(test['Date'] >= '2015-08-01') &
            (test['Date'] <= '2015-09-17')]
test = test.merge(store, on='Store')
# --

# %%
'''
Using the newly created `CompetitionDate` variable, whether the store had nearby competition at the date the data was recorded at could be determined. `CompetitionDistance` was converted into a binary variable, `NearbyCompetition`, considering any open competition nearer than the 50% quantile of all distances to be nearby. This is given by the `competition_fix` function.
'''

# %%


def competition_fix(df: pd.DataFrame) -> pd.DataFrame:
    """Consider nearby competition that is open

    Creates a binary variable to determine whether competition
    is open given the observation date. Selects competitions
    that are both open and within 10km as 1, 0 otherwise.

    Args:
        df (pd.DataFrame): Test or Train dataset in this report
    """
    # Only show competition if open by date of sale
    df['Competition'] = np.where(df['CompetitionDate'] <
                                 df['Date'], 1, 0)
    # nearby competition if open and less than 10k
    df['NearbyCompetition'] = np.where(
        (df['log_competition_dist'] > df['log_competition_dist'].quantile()) &
        (df['Competition'] == 1),
        1, 0
    )
    return df


train = competition_fix(train)
test = competition_fix(test)
# --

# %%
'''
The `average_spend` variable was created by dividing the number of sales by the number of customers. Interestingly for store type b, the daily sales were highest, but customers spent by far the least on average. See Table \ref{tab:avg}
'''

# %%
# new av spend variable
train['average_spend'] = train['Sales'] / train['Customers']

vals = {}
for storetype in train['StoreType'].unique():
    r = train[train['StoreType'] == storetype]
    vals[storetype] = r[['Sales', 'Customers', 'average_spend']].mean()

# --

# %%

# while average daily sales for b are highest, customers also spend
# by far the least
print(pd.DataFrame(vals))
# --

# %%
'''
Whether Promotion 2 was active in a particular store at the date was derived and given a binary value, given by the `promo_fix` function, holidays were also converted to binary variables. Finally, all now unneeded columns were dropped from both the test and training data.
'''

# %%


def promo_fix(df: pd.DataFrame) -> pd.DataFrame:
    # promo2 active if since week and year
    df['Promo2Precise'] = np.where(
        df['Promo2Start'] <= df['Date'], 1, 0
    )
    return df


train = promo_fix(train)
test = promo_fix(test)


def convert_holidays(df: pd.DataFrame) -> pd.DataFrame:
    df['StateHoliday'] = np.where(df['StateHoliday'] == '0', 0, 1)
    return df


train = convert_holidays(train)
test = convert_holidays(test)


def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(['log_competition_dist',
             'Promo2',
             'CompetitionDate',
             'Promo2Start',
             'Competition'], axis=1, inplace=True)
    return df


train = drop_cols(train)
test = drop_cols(test)
# --

# %%
'''
# Stationarity of TimeSeries

Timeseries data is likely to suffer from seasonality. Figures show that there is very clearly both a weekly seasonality, and a monthly seasonality.
'''

# %%


def group_dates(df, time_period):
    df.groupby([time_period, 'StoreType'],
               as_index=False)['Sales'].mean()
    return df


weekly_train = group_dates(train, 'DayOfWeek')

train['Month'] = train['Date'].dt.month
monthly_train = group_dates(train, 'Month')

sns.catplot(data=weekly_train, x='DayOfWeek', y='Sales',
            col='StoreType', kind='point', hue='StoreType')
sns.catplot(data=monthly_train, x='Month', y='Sales',
            col='StoreType', kind='point', hue='StoreType')
plt.show(block=False)
# --

# %%
'''
What about overall trend? Overall increasing trend for each storetype.
'''

# %%
# visualise overall trend between store types

train.set_index('Date', inplace=True)


def resample_by_col(df: pd.DataFrame, col: pd.Series, rule: str) -> Dict:
    return {x: df[df[col] == x]
            .resample(rule=rule)['Sales'].mean()
            for x in df[col].unique()}


resampled_storetype = resample_by_col(train, 'StoreType', '1M')
fig, ax = plt.subplots(figsize=(10, 10))
for key, storetype in resampled_storetype.items():
    ax.plot(storetype, label=key)
    ax.legend()
plt.show(block=False)
# --

# %%
'''
## Accounting for Trend and Seasonality

Seasonal decompose provides seasonal and trend values which attempt to account for these changes over time. These may then be added as variables, filling in NA values with the previous or next value.
'''

# %%

def create_trends(df, type):
    if type == 'seasonal':
        trend = {keys: sm.tsa.seasonal_decompose(storetype).seasonal
                 for keys, storetype in df.items()}
    elif type == 'trend':
        trend = {keys: sm.tsa.seasonal_decompose(storetype).trend
                 for keys, storetype in df.items()}
    trend = pd.melt(
        pd.DataFrame(trend).reset_index(),
        id_vars='Date',
        value_name=type,
        var_name='StoreType'
    )
    return trend


seasonal_trend = create_trends(resampled_storetype, type='seasonal')
time_trend = create_trends(resampled_storetype, type='trend')


# ffill will fill in gaps from start of month to end by store type
train = train.merge(seasonal_trend, on=['Date', 'StoreType'], how='left')\
    .fillna(method='ffill')

train = train.merge(time_trend, on=['Date', 'StoreType'], how='left')\
    .fillna(method='ffill')\
    .fillna(method='bfill')
# --

# %%
'''
With all the variables now created, a heatmap can now be used to determine the correlation between each. Figure \ref{fig:corr} indicates that there is a clear drop in sales as the week progresses, due to Sunday trading hours. Promo gives higher sales, holidays mean lower sales, and there may be a slight drop in sales with Promo2.
'''


# %%
fig, ax = plt.subplots(figsize=(10, 10))
corr = train.drop(['Open', 'Store'], axis=1).corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show(block=False)
# --

# %%
'''
Interestingly, average spend is negatively correlated with the time trend, suggesting that customers are spending less on average, despite customers being correlated positively with time. Other two correlations are the monthly and yearly trends, explore with fb prophet.
'''

# %%
train.to_csv('~/data/modules/envs801/derived/train_clean.csv')
test.to_csv('~/data/modules/envs801/derived/test_clean.csv')
# --

# %%
'''
# Modelling the Trend

This analysis uses the Facebook Prophet library for forecasting. Prophect uses **blah, something about linear models**.

First the libraries were loaded in along with the cleaned training and test data.
'''

# %%
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import performance_metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from typing import Tuple
from multiprocessing import Pool, cpu_count



train = pd.read_csv('~/data/modules/envs801/derived/train_clean.csv',
                    index_col=0)
test = pd.read_csv('~/data/modules/envs801/derived/test_clean.csv',
                   index_col=0)

# --

# %%
'''
For this analysis it was determined that it would be simpler to keep only stores which were open, as sales were of interest and any store closed should have no sales. Also due to the clear differences between store types, it was clear that the analysis should be split. The following section explores the prediction of the future number of sales for the 'b' type of stores only. The analysis particularly takes the total number of sales per day across all 'b' stores.
'''

# %%
# keep only open stores
# sales will reflect closed stores
train = train[train['Open'] == 1]
test = test[test['Open'] == 1]

store_types = train['StoreType'].unique()


def groupby_storetype(df: pd.DataFrame, store_type: str) -> pd.DataFrame:
    df = df[df['StoreType'] == store_type]\
        .loc[:, ['Date',
                 'Sales',
                 'Promo',
                 'Promo2Precise',
                 'Assortment_b',
                 'Assortment_c',
                 'NearbyCompetition'
                 ]]\
        .groupby(['Date'], as_index=False).sum()
    df['Date'] = pd.DatetimeIndex(df['Date'])
    df.rename(columns={'Date': 'ds',
                       'Sales': 'y'}, inplace=True)
    return df


sales = groupby_storetype(train, store_type='b')
test = groupby_storetype(test, store_type='b').drop('y', axis=1)
# --

# %%
'''
The total daily sales for the 'b' stores were plotted for reference.
'''

# %%
# plot daily sales
ax = sales.set_index('ds')['y'].plot(figsize=(12, 4))
ax.set_ylabel('Daily Number of Sales')
ax.set_xlabel('Date')
plt.show(block=False)
# --

# %%
'''
Prophet takes the holidays into account, so these were separated into their own dataframe.
'''

# %%
state_dates = train[(train['StateHoliday'] == 1)]['Date'].values
school_dates = train[train['SchoolHoliday'] == 1]['Date'].values

state = pd.DataFrame({'holiday': 'state_holiday',
                      'ds': pd.to_datetime(state_dates)})
school = pd.DataFrame({'holiday': 'school_holiday',
                       'ds': pd.to_datetime(school_dates)})

holidays = pd.concat((state, school))
# --

# %%
'''
The model could now be built.
'''

# %%


def create_model(
        df: pd.DataFrame,
        include_regressors: bool,
        holidays: pd.DataFrame,
        test_data: pd.DataFrame = None) -> Tuple[Prophet, pd.DataFrame, float]:
    """Fit Prophet models with adjustable params

    Allows for inclusion of test data, holidays, and regressors inclusion
    in Prophet model. Function will fit the model to output the model, 
    forecasted values, optionally including future dates, and the global
    root mean square error.

    Args:
        df (pd.DataFrame): Containing y output var, optionally regressors
        include_regressors (bool): Optionally include all regressors
        holidays (pd.DataFrame): Contains dates for all holidays
        test_data (pd.DataFrame, optional): future dates, optional regressors

    Returns:
        Tuple[Prophet, pd.DataFrame, int]: Model, forecast values and RMSPE.
    """
    model = Prophet(holidays=holidays,
                    daily_seasonality=False)

    if include_regressors:
        for col in df.columns[2:]:
            model.add_regressor(col)

    model.fit(df)

    if test_data is not None:
        forecast = model.predict(df.append(test_data))
    else:
        forecast = model.predict(df)

    fc = forecast.merge(df[['ds', 'y']], on='ds', how='left')
    fc_drop = fc.dropna()
    # values lower than zero = 1 for rmse calcs
    fc_drop['yhat'] = np.where(fc_drop['yhat'] <= 0, 1, fc_drop['yhat'])
    fc_drop['y'] = np.where(fc_drop['y'] <= 0, 1, fc_drop['y'])

    def find_rmspe(y_true: pd.Series, y_pred: pd.Series) -> float:
        return np.sqrt(
            np.mean(np.square(((y_true - y_pred) / y_true)), axis=0)
        ) * 100

    rmspe = find_rmspe(fc_drop['y'], fc_drop['yhat'])
    return model, fc, rmspe

# --

# %%

# smaller rmspe with regressors
model1, fc1, rmspe1 = create_model(sales,
                                   include_regressors=True,
                                   holidays=holidays)
print(rmspe1)
model2, fc2, rmspe2 = create_model(sales,
                                   include_regressors=False,
                                   holidays=holidays)
print(rmspe2)
# --

# %%
'''
With regressors gives a small improvement in the rmspe. So this model will be used for the final analysis. Figure \ref{fig:final} gives an overview of the predicted values for the training data, and the predicted new values using the test data.
'''

# %%
# model including test data
model, fc, rmspe = create_model(sales,
                                include_regressors=True,
                                holidays=holidays,
                                test_data=test)

fig, ax = plt.subplots(figsize=(16, 8))
model.plot(fc, ax=ax)
plt.axvline(x=test['ds'].iloc[0], color='k', linestyle='dashed')
plt.show(block=False)
# --

# %%
'''
Can now perform some crossvaliation to assess the quality of the model. Figure \ref{fig:mape} indicates that the mape appears to be fairly consistant at around 10% for every horizon.
'''

# %%
# prophet crossvals
df_cv = cross_validation(model1, horizon='180 days')

df_cv.head()
df_p = performance_metrics(df_cv)
df_p.head()

fig = plot_cross_validation_metric(df_cv, metric='mape')
fig.show()
# --


from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import performance_metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from typing import Tuple

train = pd.read_csv('~/data/modules/envs801/derived/train_clean.csv',
                    index_col=0)
test = pd.read_csv('~/data/modules/envs801/derived/test_clean.csv',
                   index_col=0)

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

# plot daily sales
ax = sales.set_index('ds')['y'].plot(figsize=(12, 4))
ax.set_ylabel('Daily Number of Sales')
ax.set_xlabel('Date')
plt.show(block=False)

state_dates = train[(train['StateHoliday'] == 1)]['Date'].values
school_dates = train[train['SchoolHoliday'] == 1]['Date'].values

state = pd.DataFrame({'holiday': 'state_holiday',
                      'ds': pd.to_datetime(state_dates)})
school = pd.DataFrame({'holiday': 'school_holiday',
                       'ds': pd.to_datetime(school_dates)})

holidays = pd.concat((state, school))


def create_model(
        df: pd.DataFrame,
        include_regressors: bool,
        holidays: pd.DataFrame,
        test_data: pd.DataFrame = None) -> Tuple[Prophet, pd.DataFrame, int]:
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

    def rmspe(y_true, y_pred):
        return np.sqrt(
            np.mean(np.square(((y_true - y_pred) / y_true)), axis=0)
        ) * 100

    rmspe = rmspe(fc_drop['y'], fc_drop['yhat'])
    return model, fc, rmspe


# smaller rmspe with regressors
model1, fc1, rmspe1 = create_model(sales,
                                   include_regressors=True,
                                   holidays=holidays)
print(rmspe1)
model2, fc2, rmspe2 = create_model(sales,
                                   include_regressors=False,
                                   holidays=holidays)
print(rmspe2)
# model including test data
model, fc, rmspe = create_model(sales,
                                include_regressors=True,
                                holidays=holidays,
                                test_data=test)

fig, ax = plt.subplots(figsize=(16, 8))
model.plot(fc, ax=ax)
plt.axvline(x=test['ds'].iloc[0], color='k', linestyle='dashed')
plt.scatter(fc[fc['y'].isna()]['ds'], fc[fc['y'].isna()]['yhat'], color='red',
            s=10)
plt.show(block=False)

# prophet crossvals
df_cv = cross_validation(model1, horizon='180 days')

df_cv.head()
df_p = performance_metrics(df_cv)
df_p.head()

fig = plot_cross_validation_metric(df_cv, metric='mape')
fig.show()

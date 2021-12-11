import pandas as pd

from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits = load_digits()
digits.target

data = pd.read_csv('/home/cjber/data/modules/envs801/derived/train_clean.csv',
                   index_col=0,
                   parse_dates=['Date'])\
    .set_index(['Date'])
data = data\
    .drop(['StoreType', 'Store', 'DayOfWeek', 'Month', 'Open'], axis=1)\
    .groupby('Date', as_index=False).sum()

X_train, X_test, y_train, y_test = train_test_split(data.drop('Sales', axis=1),
                                                    data['Sales'])

pipeline_optimizer = TPOTRegressor(generations=5, population_size=20,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))

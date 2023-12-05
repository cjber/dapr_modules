# %%
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import exp, log1p, sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

base = pd.read_csv("./data/derived/cars_01_cleaned.csv", index_col=0)
res = pd.read_csv("./data/derived/lm_results.csv", index_col=0)
df = base.join(res)

df.columns
cont_vars = ['price', 'powerPS', 'age', 'kilometer']

f, axs = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()
for i, m in enumerate(res.columns.drop(["Truth", "Truth-Log"])):
    ax = axs[i]
    sns.regplot("Truth",
                m,
                res,
                ci=None,
                ax=ax
                )
f.suptitle("Observed Vs Predicted")
plt.show()

r2s = pd.Series({"LM-Raw": r2_score(df["Truth"],
                                    df["LM-Raw"]
                                    ),
                 "LM-Log": r2_score(log1p(df["Truth"]),
                                    df["LM-Log"]
                                    ),
                 "RF-Raw": r2_score(df["Truth"],
                                    df["RF-Raw"]
                                    ),
                 "RF-Log": r2_score(log1p(df["Truth"]),
                                    df["RF-Log"]
                                    ),
                 })
r2s

mses = pd.Series({"LM-Raw": mse(df["Truth"],
                                df["LM-Raw"]
                                ),
                  "LM-Log": mse(df["Truth"],
                                exp(df["LM-Log"])
                                ),
                  "RF-Raw": mse(df["Truth"],
                                df["RF-Raw"]
                                ),
                  "RF-Log": mse(df["Truth"],
                                exp(df["RF-Log"])
                                ),
                  }).apply(sqrt)
mses

# %%
"""
# Cross Validation
"""


x_train, x_test, y_train, y_test = train_test_split(df[cont_vars],
                                                    df['price'],
                                                    test_size=0.8)

lm_estimator = LinearRegression()
lm_estimator.fit(x_train, y_train)
lm_y_pred = lm_estimator.predict(x_test)
sqrt(mse(y_test, lm_y_pred))


lm_kcv_mses = cross_val_score(LinearRegression(),
                              df[cont_vars],
                              df["price"],
                              cv=5,
                              scoring="neg_mean_squared_error"
                              )
# sklearn uses neg to optimisation is alway maximisation
sqrt(-lm_kcv_mses).mean()

rf_estimator = RandomForestRegressor(n_estimators=100,
                                     max_features=None
                                     )
rf_kcv_mses = cross_val_score(rf_estimator,
                              df[cont_vars],
                              df["price"],
                              cv=5,
                              scoring="neg_mean_squared_error"
                              )
# sklearn uses neg to optimisation is alway maximisation
sqrt(-rf_kcv_mses).mean()

mses  # difference in rf mses, evidence of overfitting

# %%
"""
# Parameter optimisation
"""

param_grid = {
    "n_estimators": [5, 25, 50, 75, 100, 150],
    "max_features": [1, 2, 3, 4]
}


grid = GridSearchCV(RandomForestRegressor(),
                    param_grid,
                    scoring="neg_mean_squared_error",
                    cv=5,
                    n_jobs=-1
                    )

grid.fit(df[cont_vars], df['price'])
grid_res = pd.DataFrame(grid.cv_results_)
grid_res[grid_res["mean_test_score"] ==
         grid_res["mean_test_score"].max()
         ]["params"]

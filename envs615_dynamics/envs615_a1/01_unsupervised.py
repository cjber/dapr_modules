# %%
from typing import List
import seaborn as sns
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

"""
# Predicting Car Listing Value through Machine Learning

This essay presents the results of a python based machine learning exercise in which car priceses scraped from Ebay listings are explored.
"""

# %%
df = pd.read_csv("./data/derived/cars_cleaned.csv", index_col=0)
cont_vars = ["price",
             "age",
             "powerPS",
             "kilometer"]
df[cont_vars].hist()
plt.show()

# %%
"""
Price skew, power age more normal, kilometer shows gaps.
"""

# %%
cat_vars = [
    "vehicleType",
    "gearbox",
    "fuelType",
    "brand",
    "area"
]

df[cat_vars].describe().T  # model too many unique values?

# %%
"""
Large number of categories with area and brand, likely that some may be dropped if they contain few values.
"""

# %%
# remove categoric values that have a very low number of cols
df = df[df['fuelType'].isin(
    df["fuelType"]
    .value_counts()
    .head(2)  # for fuel best to keep only petrol and diesel
    .index
    .tolist()
)]

# keep only top 5 categories by count for the rest of the categoric vars
for var in cat_vars:
    df = df[df[var].isin(
        df[var]
        .value_counts()
        .head(5)  # for the rest we can keep 5
        .index
        .tolist())]

# %%
"""
Kilometer has weird gaps:
"""
# %%
df['kilometer'].unique()  # more like an ordinal variable

# %%
"""
very few unique values so more like an ordinal variable.
"""
# %%
df = df[df['kilometer'].isin(
    df["kilometer"]
    .value_counts()
    .head(5)
    .index
    .tolist()
)]

df['kilometer'] = df.loc[:, 'kilometer']\
    .astype(CategoricalDtype(ordered=True))

# fix the cont and cat vars lists
cont_vars.remove("kilometer")
cat_vars.append("kilometer")
# %%
"""
# Unsupervised Learning

First a K Means clustering attempt. Determine optimal number of clusters based on Calinski Harabasz Score.
"""


# %%
"""
5 clusters appears optimal with the raw dataset.
"""
# %%

df_std = scale(df[cont_vars])
df_std = pd.DataFrame(df_std, index=df.index, columns=cont_vars)

range_scaler = MinMaxScaler()
df_scaled = range_scaler.fit_transform(df[cont_vars])
df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=cont_vars)

plts = {}
itr = 0
df_norms = [df, df_std, df_scaled]

for d in df_norms:
    chss = {}
    for i in range(2, 11):
        estimator = KMeans(n_clusters=i)
        estimator.fit(d[cont_vars])
        chs = calinski_harabasz_score(d[cont_vars],
                                      estimator.labels_)
        chss[i] = chs
    plts[itr] = chss
    itr += 1
plts = [pd.Series(plts[k]) for k in plts.keys()]

f, ax = plt.subplots(1, 3, figsize=(16, 6))
[plts[i].plot.line(ax=ax[i]) for i, _ in enumerate(plts)]


def annotate_axes(f, title, x_lab, y_lab):
    for i, ax in enumerate(f.axes):
        ax.tick_params(labelbottom=x_lab, labelleft=y_lab)
        ax.set_title(title[i])


annotate_axes(f, title=['Raw', 'Std', 'Scaled'], x_lab=True, y_lab=False)
plt.show()

# %%
"""
Figure shows that using the unmodified dataframe that there appears to be 5-7 groups.

Comparision between models.
"""
# %%
# initial number of clusters
estimator = KMeans(n_clusters=6)


def k5_fits(df):
    estimator.fit(df[cont_vars])
    k5_fits = pd.Series(estimator.labels_, index=df.index)
    return k5_fits


# fit raw/std/scaled model
k5_fits = [k5_fits(df) for df in df_norms]

pca_estimator = PCA(n_components=2)


def k5_pca(df):
    components = pca_estimator.fit_transform(df[cont_vars])
    components = pd.DataFrame(components,
                              index=df.index,
                              columns=["C-1", "C-2"])
    estimator.fit(components)
    k5_pca = pd.Series(estimator.labels_, index=components.index)
    return k5_pca


df_pca = [df_std, df_scaled]
# fit pca_std/pca_scaled
k5_pca = [k5_pca(df) for df in df_pca]

k5_fits.extend(k5_pca)


def plt_components(labs, ax):
    # choose std df components for visualisations
    components = pca_estimator.fit_transform(df_std[cont_vars])
    components = pd.DataFrame(components,
                              index=df.index,
                              columns=["C-1", "C-2"])
    components.assign(labels=labs)\
        .plot.scatter("C-1",
                      "C-2",
                      c="labels",
                      s=1,
                      cmap="Paired",
                      colorbar=False,
                      ax=ax)


f = plt.figure(figsize=(12, 6))
ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
axs = [ax1, ax2, ax3, ax4, ax5]
# plot
for i, labs in enumerate(k5_fits):
    plt_components(labs, axs[i])
annotate_axes(f, title=['Raw', 'Std', 'Scaled', 'PCA-Std', 'PCA-Scaled'],
              x_lab=False, y_lab=False)
plt.show()

# %%
"""
From these plots PCA-std seems most appropriate.

# Explore Classification

Calinski Harabasz Score.
"""

# %%
# explore classification
chs = [calinski_harabasz_score(df[cont_vars], k5) for k5 in k5_fits]
pd.Series({"Raw": chs[0],
           "Standardised": chs[1],
           "Scaled": chs[2],
           "PCA Standardised": chs[3],
           "PCA Scaled": chs[4],
           })

# %%
"""
While the score for the raw values is much higher, but all variables are at different scales. From this is appears that the non-PCA standardised provides the best score.
"""

# %%


def sil_scores(k5_fits):
    sil_scores = silhouette_score(
        df[cont_vars],
        k5_fits,
        metric="euclidean",
        sample_size=10000
    )
    return sil_scores


sil_score = [sil_scores(fit) for fit in k5_fits]
pd.Series({"Raw": sil_score[0],
           "Standardised": sil_score[1],
           "Scaled": sil_score[2],
           "PCA Standardised": sil_score[3],
           "PCA Scaled": sil_score[4]})

# %%

g = df_std[cont_vars]\
    .groupby(k5_fits[1])
sns.heatmap(g.mean(), cmap='viridis')
plt.show()

df_scaled.groupby(k5_fits[1])[["price", "age"]]\
    .mean()\
    .sort_values('age')['price']\
    .plot.barh(color="orange",
               alpha=.5)
plt.show()

pd.DataFrame({"k5_std": k5_fits[1]})\
    .to_csv("./data/derived/k5_std.csv")
df.to_csv("./data/derived/cars_01_cleaned.csv")

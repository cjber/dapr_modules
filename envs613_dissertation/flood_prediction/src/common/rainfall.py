from pathlib import Path
import geopandas as gpd
import pandas as pd

import jsonlines
import requests
from tqdm import tqdm

readings_path = Path("data/floods")

r = requests.get(
    "http://environment.data.gov.uk/flood-monitoring/id/stations?_limit=10000"
)
stations = pd.DataFrame(r.json()["items"])
stations = stations[["stationReference", "lat", "long"]].dropna()
stations = gpd.GeoDataFrame(
    stations, geometry=gpd.points_from_xy(stations["long"], stations["lat"])
)

with jsonlines.open("data/floods/flood_tweets.jsonl", "r") as jl:
    tweets = pd.DataFrame([tweet for tweet in jl])

dates = tweets["created_at"].str[:-14].unique().tolist()


# with open(readings_path / "readings.csv", "wb") as fd:
#     for date in tqdm(dates):
#         r = requests.get(
#             f"http://environment.data.gov.uk/flood-monitoring/archive/readings-full-{date}.csv",
#             stream=True,
#         )
#         if r.status_code == 200:
#             for chunk in r.iter_content(chunk_size=10_000):
#                 fd.write(chunk)


reading = pd.read_csv(readings_path / "readings.csv")
reading = reading.merge(stations, on="stationReference")
reading = reading[["dateTime", "value", "stationReference", "geometry"]].dropna()
reading["date"] = pd.to_datetime(reading["dateTime"].str[:-10])
reading["value"] = reading["value"].str.split("|", expand=True)[0].astype(float)
reading = gpd.GeoDataFrame(reading, geometry="geometry")

flood_areas = gpd.read_file("data/floods/flood_areas.gpkg")
flood_areas = flood_areas[["fwdCode", "geometry"]].dropna()
stations_by_area = gpd.sjoin(stations, flood_areas, how="inner", op="intersects")

reading = reading.merge(stations_by_area, on="stationReference")

flood_areas["bounding_box"] = flood_areas.geometry.bounds.apply(
    lambda x: str(list(x)), axis=1
)
tweets["bounding_box"] = tweets["bounding_box"].astype(str)
tweets = tweets.merge(flood_areas, on="bounding_box")

tweets["date"] = pd.to_datetime(tweets["created_at"].str[:-14])
tweets["warning_time"] = pd.to_datetime(tweets["warning_time"].str[:-9])
tweets = tweets.sort_values("created_at")
tweets["diff_date"] = tweets["date"] - tweets["warning_time"]

tweets = tweets.merge(reading, on=["date", "fwdCode"])

tweets = (
    tweets[["date", "idx", "value", "diff_date"]]
    .dropna()
    .groupby(["idx", "date", "diff_date"])
    .mean()
)

tweets.to_csv("data/floods/rainfall_dates.csv")

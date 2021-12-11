from pathlib import Path

import pandas as pd



def pre_labelling(tweets):
    tweets_label = tweets.sample(2_000, random_state=42)
    return tweets_label[["text", "label"]]


if __name__ == "__main__":
    tweets = pd.read_csv("data/floods/flood_tweets.csv")
    pre_label_dir = Path("data/train/to_label.csv")

    if not pre_label_dir.exists():
        tweets_label = pre_labelling(tweets)
        tweets_label[["text", "label"]].to_csv(pre_label_dir, index=False)

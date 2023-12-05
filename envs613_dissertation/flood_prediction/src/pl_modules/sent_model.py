import jsonlines
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from src.common.utils import preprocess

with jsonlines.open("data/floods/flood_tweets.jsonl", "r") as jl:
    tweets = pd.DataFrame(list(jl))


def preprocess_tweets(tweets: pd.DataFrame) -> dict[int, pd.DataFrame]:
    tweets["created_at"] = pd.to_datetime(tweets["created_at"].str[:-14])
    tweets["warning_time"] = pd.to_datetime(tweets["warning_time"].str[:-9])

    tweets = tweets.sort_values("created_at")
    tweets["diff_date"] = tweets["created_at"] - tweets["warning_time"]
    tweets = tweets.groupby("idx").filter(lambda x: len(x) > 500)

    return dict(tuple(tweets.groupby("idx")))


def build_model(task: str):
    # Tasks:
    # emoji, emotion, hate, irony, offensive, sentiment
    # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.to("cuda")
    return model, tokenizer


if __name__ == "__main__":
    tweets_idx = preprocess_tweets(tweets)
    model, tokenizer = build_model(task="sentiment")

    sent_dict = []
    for idx, event in tqdm(tweets_idx.items()):
        for idx, row in event.iterrows():
            encoded = tokenizer(
                preprocess(row["text"]),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128,
            ).to("cuda")

            output = model(**encoded, output_attentions=True)

            attn = output["attentions"][-1].squeeze()[-1].cpu().detach().numpy()
            score = F.softmax(output["logits"], dim=1).cpu().detach().numpy().flatten()

            sent_dict.append(
                {
                    "idx": idx,
                    "negative": score[0],
                    "neutral": score[1],
                    "positive": score[2],
                    "diff_date": row["diff_date"],
                }
            )

    pd.DataFrame(sent_dict).to_csv("data/floods/flood_sent.csv", index=False)

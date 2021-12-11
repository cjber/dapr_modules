import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from src.common.utils import Const, combine_biluo, combine_subwords

model = AutoModelForTokenClassification.from_pretrained(
    "cjber/geographic-entity-recognition"
)
tokenizer = AutoTokenizer.from_pretrained(
    "roberta-base", max_length=128, padding="max_length", truncation=True
)
ner = pipeline("ner", model=model, tokenizer=tokenizer, device=0)  # type: ignore
flood_tweets = pd.read_csv("data/out/full_labelled.csv")


def get_places(row):
    out = ner(inputs=row["text"])
    try:
        words = [word["word"] for word in out]
        tags = [tag["entity"] for tag in out]

        words, tags = combine_subwords(words, [Const.LABELS[idx] for idx in tags])
        words, tags = combine_biluo(words, tags)
    except ValueError:
        return None
    return words


tqdm.pandas()
flood_tweets["places"] = flood_tweets.progress_apply(lambda x: get_places(x), axis=1)

flood_tweets.to_csv("data/out/flood_places.csv", index=False)

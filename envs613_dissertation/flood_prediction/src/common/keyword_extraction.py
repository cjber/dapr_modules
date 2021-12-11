from keybert import KeyBERT

doc = """
    A car is floating down the road because of floods.
      """

model = KeyBERT("distilbert-base-nli-mean-tokens")
keywords = model.extract_keywords(doc)

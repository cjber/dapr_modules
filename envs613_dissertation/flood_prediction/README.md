<div align="center">

# Using Transformers to Extract Relevant Information From Twitter During Flood Events (Dissertation)

<a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white"/></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>

</div>

[Cillian
Berragan](https://www.liverpool.ac.uk/geographic-data-science/our-people/)
\[[`@cjberragan`](http://twitter.com/cjberragan)\]<sup>1\*</sup>

<sup>1</sup> *Geographic Data Science Lab, University of Liverpool,
Liverpool, United Kingdom*

<sup>*</sup> *Correspondence\*: C.Berragan@liverpool.ac.uk

## Abstract

Social media presents a source of real-time information provided by
individual users in emergency situations. However, due to its
unstructured nature and high volume, key information is challenging to
extract from these continuous data streams. My dissertation considers
the ability to identify relevant flood related Tweets from a Twitter
corpus, extracted during dates relating to a historic archive of past
severe flooding events in the United Kingdom. A deep neural
classification model is constructed, built on the transformer
architecture with a pre-trained language model base. My dissertation
demonstrates that this model outperforms both simple rule-based
approaches, and past machine learning methodologies, often used in past
literature. Following classification, spatio-temporal analysis is
performed to observe how information through Tweets develops over
emergency flood events.

## Description

This repository contains the code for building a DistilBERT-based binary
classification model, trained to identify relevant and irrelevant flood
related Tweets. Model training uses a labelled corpus of Tweets
extracted during past severe flood events in the United Kingdom, using
flood zone bounding boxes.

Inference over a separate testing corpus is compared against a keyword
based classification method.

## Project layout

``` bash
src
├── common
│   ├── get_tweets.py  # download tweets to csv through twitter api
│   └── utils.py  # various utility functions
│
├── datasets
│   ├── csv_dataset.py  # torch dataset for flood data
│   └── datamodule.py  # lightning datamodule
│
├── modules
│   └── classifier_model.py  # flood classification model
│
├── run.py  # train model
└── inf.py  # use model checkpoint for inference and compare with keywords
```

## How to run

> Note: At present the Tweet corpus is not available for model training

### Poetry

Install dependencies using [Poetry](https://python-poetry.org/):

``` commandline
poetry install
```

Train classifier model using the labelled flood Tweets corpus:

``` commandline
poetry run python -m src.run
```

### Docker

Build image from Dockerfile:

``` bash
docker build . -t cjber/flood_tweets
```

Run with GPU and mapped volumes:

``` bash
docker run --rm --gpus all -v ${PWD}/ckpts:/flood/ckpts -v ${PWD}/csv_logs:/flood/csv_logs cjber/flood_tweets
```

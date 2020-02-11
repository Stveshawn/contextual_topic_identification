# Contextual Topic Identification for Steam Reviews

This repository is the implementation of contextual topic identification model. The model is based on _LDA_ probabilistic topic assignment and pre-trained sentence embeddings from _BERT/RoBERTa_. The analysis is conducted on the dataset of game reviews on the steam platform.

## Motivation

Product reviews are important as influencing people's choices especially for online shopping. We usually have dreadfully huge numbers of reviews for whatever products. However, many platforms have barely a satisfying categorization system for the reviews when it comes to what the reviewers are really talking about. Steam, for example, has a very carefully designed system where people can do a lot of things, but still there is no such access to categorizing the reviews by their semantic meanings.

![Steam review logo](./docs/images/steam_review.jpeg)

Therefore, we provide a topic identification procedure thats combines both bag-of-words and contextual information to explore potential __semantically meaningful__ categories out of the oceans of steam reviews.

## Setup

Clone the repo

```
git clone https://github.com/Stveshawn/contextual_topic_identification.git
cd contextual_topic_identification
```

and make sure you have dataset in the `data` folder (you can specify the path in the bash script later).


To run the model and get trained model objects and visualization

### With Docker

run the bash script on your terminal

```
sudo bash docker_build_run.sh
```

The results will be saved in the `docs` folder with corresponding model id (_Method_Year_Month_Day_Hour_Minute_Second_).

Four parameters can be specified in the bash script

+ `samp_size`: number of reviews used in the model
+ `method={"TFIDF", "LDA", "BERT", "LDA_BERT"}`: method for the topic model
+ `ntopic`: number of topics
+ `fpath=/contextual_topic_identification/data/steam_reviews.csv`: file path to the csv data




## Data

The data used ([Steam review dataset](https://www.kaggle.com/luthfim/steam-reviews-dataset)) is published on Kaggle covering ~480K reviews for 46 best selling video games on steam.

## Pipeline

+ steam review texts

+ preprocessing
  + lowercase
  + normalization (language, repetition)
  + lemmatization
  + stop words
  + fix typos

+ doc embedding: word embedding weighted by TF-IDF

+ downstream tasks

  + clustering
  + detecting meaningless reviews

## Result

webapp by streamlit

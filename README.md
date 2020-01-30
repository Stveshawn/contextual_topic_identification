# Topic Identification based on Sentence Embedding (TISE) for steam reviews

This repository is the implementation of semantically meaningful topic identification by sentence embedding. The implementation is based on pre-trained sentence embedding models (BERT/RoBERTa). The analysis is conducted on the dataset of game reviews on the steam platform.

## Motivation

Product reviews are important as influencing people's choices especially for online shopping. We usually have dreadfully huge numbers of reviews for whatever products. However, many platforms have barely a satisfying categorization system for the reviews when it comes to what the reviewers are really talking about. Steam, for example, has a very carefully designed system where people can do a lot of things, but still there is no such access to categorizing the reviews by their semantic meanings.

![Steam review logo](https://steamcdn-a.akamaihd.net/steam/clusters/about_i18n_assets/about_i18n_assets_0/feature_reviews_header_english.jpg?t=1569333767)

Therefore, we provide a topic identification procedure based on sentence embedding and unsupervised learning methods to explore __semantically meaningful__ categories out of the oceans of steam reviews.

## Setup

### Installation

+ Instruction: run main.py to get plots from clustering. It will call the other functions. (Data and modules are required to run the code. Docker to be added in week 3.)

+ Dependency: downstream.py -> embedding.py -> preprocessing.py

## Data

[Steam review dataset](https://www.kaggle.com/luthfim/steam-reviews-dataset)

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

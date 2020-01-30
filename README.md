# Topic Identification based on Sentence Embedding (TISE) for product reviews

This repository is the implementation of semantically meaningful topic identification by sentence embedding. The implementation is based on pre-trained sentence embedding models (BERT/RoBERTa). The analysis is conducted on the dataset of game reviews on the steam platform.

## Motivation

Product reviews are important as influencing people's choices especially for online shopping. We usually have dreadfully huge numbers of reviews for whatever products. However, many platforms have barely a satisfying categorization system for the reviews when it comes to what the reviewers are really talking about. Steam, for example, has a very carefully designed system where people can do a lot of things, but still there is no such access to categorizing the reviews by their semantic meanings.

Therefore, we provide a topic identification procedure based on sentence embedding and unsupervised learning methods to explore semantically meaningful categories out of the oceans of steam reviews.



## Installation

+ Instruction: run downstream.py to get plots from clustering. It will call the other two functions. (Data and modules are required to run the code. Docker to be added in week 3.)

+ Dependency: downstream.py -> embedding.py -> preprocessing.py

## Motivation

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

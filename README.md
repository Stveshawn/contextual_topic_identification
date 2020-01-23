# Steam_Review_Text_Embedding
Steam review texting embedding analysis

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

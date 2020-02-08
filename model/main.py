from model import *
from utils import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default='/contextual_topic_identification/data/steam_reviews.csv')
    parser.add_argument('--ntopic', default=10)
    parser.add_argument('--method', default='TFIDF')
    parser.add_argument('--samp_size', default=10000)
    args = parser.parse_args()

    data = pd.read_csv(str(args.fpath))
    data = data.fillna('')  # only the comments has NaN's
    rws = data.review
    sentences, token_lists, idx_in = preprocess(rws, samp_size=int(args.samp_size))
    # Define the topic model object
    tm = Topic_Model(k = int(args.ntopic), method = str(args.method))
    # Fit the topic model by chosen method
    tm.fit(sentences, token_lists)
    # Evaluate using metrics
    with open("/contextual_topic_identification/docs/saved_models/{}.file".format(tm.id), "wb") as f:
        pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)

    print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
    print('Silhouette Score:', get_silhouette(tm))
    # visualize and save img
    visualize(tm)
    for i in range(tm.k):
        get_wordcloud(tm, token_lists, i)

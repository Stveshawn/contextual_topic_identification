# Word Embedding
from preprocess import *
import numpy as np
import pandas as pd
import itertools
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')





def embedding(fpath, pct, pre_trained='roberta-base-nli-stsb-mean-tokens'):
    """
    :param fpath: path/to/data.csv
    :param pct: : percentage used out of the whole dataset
    :param pre_trained: pretrained model in sentence embedding
    :return: (n_sen * d_embedding). Sentence embeddings, each review might have multiple sentences
    """

    data = pd.read_csv(fpath)
    data = data.fillna('')  # only the comments has NaN's
    print("Loading Pre-trained sentence embeddings ...")
    model = SentenceTransformer(pre_trained)
    rws = data.review.values
    n = len(rws)
    n_entry = int(pct * n)
    print("Loading Pre-trained sentence embeddings done.")

    rws_processed = []
    idx_in = []
    for i, rw in enumerate(rws[:n_entry]):
        rw_processed = preprocess(rw)
        if rw_processed:
            idx_in.append(i)
            rws_processed.append((rw_processed))
        print('{} %'.format(str(np.round(i/n_entry*100,2))), end='\r')


    # corpus = list(map(lambda x: ' '.join(x), rws_processed))
    corpus = list(itertools.chain(*rws_processed))
    sentence_embeddings = model.encode(corpus, show_progress_bar=True)
    vecs = np.array(sentence_embeddings)
    return vecs
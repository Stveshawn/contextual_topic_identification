# Word Embedding
from preprocess import *
import gensim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

model = gensim.models.KeyedVectors.load_word2vec_format('../steam-reviews-dataset/enwiki_20180420_100d.txt', binary=False, limit=50000)


# fpath = sys.argv[1]
fpath = '../steam-reviews-dataset/steam_reviews.csv'
data = pd.read_csv(fpath)
data = data.fillna('') # only the comments has NaN's
rws = data.review.values

# extra features
rws_len = np.array(list(map(len, rws)))
h = data.helpful.values

rws_processed = []
for i, rw in enumerate(rws[:20000]):
    rw_processed = preprocess(rw)
    if rw_processed:
        rws_processed.append((rw_processed))
    print('{} %'.format(str(np.round(i/20000*100,2))), end='\r')


# using tfidf to get weights
corpus = list(map(lambda x: ' '.join(x), rws_processed))
tfidf = TfidfVectorizer()
tfidf.fit(corpus)

w_tfidf = {}
k = np.array(tfidf.get_feature_names())
v = tfidf.fit_transform(corpus).toarray()
for i, k_ in enumerate(k):
    w_tfidf[k_] = v[:,i]



def get_embedding(ws, i):
    """
    get embedding from words weighted by tfidf
    """
    r = np.zeros(100)
    ct = 0
    for w in ws:
        try:
            wt = w_tfidf[w][i]
            r += wt * model[w]
            ct += wt
        except:
            pass

    return r / ct if ct > 0 else r

vecs = np.zeros((len(rws_processed), 100))
for i, ws in enumerate(rws_processed):
    vecs[i] = get_embedding(ws, i)


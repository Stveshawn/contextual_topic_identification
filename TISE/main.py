from preprocess import *
# from embedding import *
# from downstream import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default='~/Downloads/steam-reviews-dataset/steam_reviews.csv')
    parser.add_argument('--pct', default=0.0001)
    args = parser.parse_args()

    data = pd.read_csv(args.fpath)
    data = data.fillna('')  # only the comments has NaN's
    rws = data.review.values[:300]
    vecs = preprocess(rws[0])
    print(vecs)
    # downstream(vecs)
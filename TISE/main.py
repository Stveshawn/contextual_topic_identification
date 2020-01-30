from embedding import *
from downstream import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default='~/Downloads/steam-reviews-dataset/steam_reviews.csv')
    parser.add_argument('--pct', default=0.0001)
    args = parser.parse_args()


    vecs = embedding(args.fpath, args.pct)
    downstream(vecs)
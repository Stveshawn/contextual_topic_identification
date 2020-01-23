from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from embedding import *

from sklearn.mixture import GaussianMixture

# pca
pca = PCA(n_components=2)
DR2_pca = pca.fit_transform(vecs)
plt.plot(*DR2_pca.T, '.')

# tsne
tsne = TSNE()
DR2 = tsne.fit_transform(vecs)
plt.plot(*DR2.T,'.')

# GMM
GMM = GaussianMixture(n_components=5)
GMM.fit(vecs)

lbs = GMM.predict(vecs)
plt.scatter(*DR2.T, c=lbs)
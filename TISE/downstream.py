from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
from embedding import *

from sklearn.mixture import GaussianMixture

# dimension reduction
# PPA+PCA
def downstream(vecs, dim_z=200):
    """
    show the visualization of clustering
    :param vecs: embedded vectors from sentences
    :return: None
    """
    pca = PCA(n_components=dim_z)
    pca.fit(vecs)
    print('Variation explained: {}'.format(sum(pca.explained_variance_ratio_)))
    v_dr = pca.transform(vecs)

    # GMM for clustering
    GMM = GaussianMixture(n_components=5)
    GMM.fit(v_dr)
    lbs = GMM.predict(v_dr)

    # visualization
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(v_dr)
    plt.figure(figsize=(15,10))
    plt.scatter(embedding[:,0], embedding[:,1], c=lbs, alpha=0.3)
    plt.colorbar()
    plt.show();
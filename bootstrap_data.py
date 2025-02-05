import numpy as np
import scipy
from modAL.density import information_density
from modAL.utils import multi_argmax
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from sklearn.mixture import GaussianMixture

from ER_MULTI_PRO.TwinSVC import TwinSVC
from cte import N_INITIAL, BOOTSTARP_STRATEGY, DATA_BASE


def TwinSVMClusteringData(X):
    if True:
        return np.load(f'datasets/cache/{DATA_BASE}_{BOOTSTARP_STRATEGY}.npy',allow_pickle=True)
    classifier = TwinSVC(n_clusters=N_INITIAL, init='gmm', rectangular_kernel=2., gamma='scale')
    classifier.fit(X)
    bootstrap_idx = classifier._medoids,classifier._boundry
    return bootstrap_idx


applying_PCA_function = True


def applying_PCA(x_train, x_test):
    pca = KernelPCA(n_components=3, kernel='rbf')
    x1 = pca.fit_transform(x_train)
    x2 = pca.transform(x_test)
    return x1, x2


def get_bootstrap_data(x_train, x_test):
    if N_INITIAL == 0 or BOOTSTARP_STRATEGY == 'noBoot':
        return []
    print(f'BOOTSTARP_STRATEGY:{BOOTSTARP_STRATEGY}')
    x, x_t = x_train, x_test
    if applying_PCA_function:
        x, x_t = applying_PCA(x_train, x_test)
    return TwinSVMClusteringData(x)

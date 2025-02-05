import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture as GMM
from numpy import linalg as LA
import torch
from torch import nn

from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import connected_components

from data_reading import get_data


def nearest_neighbors_graph(distance_matrix, p):
    # Initialize the graph with zeros
    n_samples = distance_matrix.shape[0]
    graph = np.zeros((n_samples, n_samples))
    # Find the k nearest neighbors for each sample
    for i in range(n_samples):
        # Get the indices of the k nearest neighbors (excluding the point itself)
        nearest_indices = np.argsort(distance_matrix[i])[1:p+1]
        # Set the graph entries to 1 for the k nearest neighbors
        graph[i, nearest_indices] = 1
        graph[nearest_indices, i] = 1
    return graph

def fold_clusters(distance_matrix, labels, n_clusters):
    min_dist, min_s1, min_s2 = float('inf'), None, None
    for s1 in range(n_clusters):
        for s2 in range(s1+1, n_clusters):
            cdists = distance_matrix[labels==s1, :][:, labels==s2]
            d1 = np.max(np.min(cdists, axis=1))
            d2 = np.max(np.min(cdists, axis=0))
            dist = max(d1, d2)
            if dist < min_dist:
                min_s1, min_s2, min_dist = s1, s2, dist
    labels[labels == min_s2] = min_s1
    labels[labels == (n_clusters - 1)] = min_s2
    return labels

def nng_clustering(X, k, p):
    # Calculate pairwise distances
    dist_matrix = distance_matrix(X, X, p=2)
    nn_graph = nearest_neighbors_graph(dist_matrix, p)
    n_components, labels = connected_components(
            csgraph=nn_graph, directed=False, return_labels=True)
    if n_components < k:
        while n_components < k:
            i, j = np.unravel_index(np.argmax(dist_matrix * nn_graph), nn_graph.shape)
            nn_graph[i, j] = 0
            nn_graph[j, i] = 0
            n_components, labels = connected_components(
                    csgraph=nn_graph, directed=False, return_labels=True)
    if n_components > k:
        while n_components > k:
            labels = fold_clusters(dist_matrix, labels, n_components)
            n_components -= 1
    return labels

def solve_torch(H, G, C,C1, lambda_, eps=1e-4, lr=2e-5, max_iter=1000, reduction_steps=800):
    H = torch.as_tensor(H).float()
    G = torch.as_tensor(G).float()
    def obj(z, eta):
        return (
                (H @ z).norm(p=1.)
                + (C*eta).sum()
                - lambda_ * ((G @ z).abs() + eta - 1).clip(max=0.).sum()
                - lambda_ * eta.clip(max=0.).sum()
                + C1*z[:-1].norm(p=1.)
                )
    z_param = nn.Parameter(torch.randn(H.shape[-1])*1e-3)
    eta_param = nn.Parameter(torch.zeros(G.shape[0]))
    for i in range(max_iter):
        loss = obj(z_param, eta_param)
        loss.backward()
        with torch.no_grad():
            z_param -= lr * z_param.grad
            eta_param -= lr * eta_param.grad
        z_param.grad.zero_()
        eta_param.grad.zero_()
        if (i + 1)%reduction_steps == 0:
            lr *= .10
    return z_param.detach().numpy()

class TwinSVC():
    def __init__(
            self,
            n_clusters=8, # Number of clusters to create
            *,
            C=10., # The penalty coefficient for the objective function
            kernel='rbf', # The kernel type: linear, polynomial, rbf, or sigmoid
            degree=2, # The degree for the polynomial kernel
            gamma='auto', # The gamma coefficient for non-linear kernels: scale, auto, or a positive float
            max_iter=100, # Maximum iterations for the fit function
            tol_alpha=1e-4, # Tolerance for the optimization steps
            tol_beta=.05, # Tolerance for the optimization steps
            init='random', # Initialization, either random, kmeans, or gmm
            coef0=0., # The bias for the polynomial and sigmoid kernels
            initial_guess='random', # The initial solution for the optimization problem: random, or eig
            num_boundry=None,
            rectangular_kernel=None,
            lambd = 1e-4,
            tau = 2e-3,
            tau1 = 1e-3,
            tau2 = 1e-3,
            a = 1e-3,
            r = 1e-3
            ):
        self.n_clusters = n_clusters
        self.C = C
        self.C1 = 10.
        self.max_iter = max_iter
        self.max_outer_iter = 10
        self.degree = degree
        self.gamma = gamma
        self.tol_alpha = tol_alpha
        self.tol_beta = tol_beta
        self.init = init
        self.kernel = kernel
        self.coef0 = coef0
        self.initial_guess = initial_guess
        self.n_features = None
        self.weights = None
        self.biases = None
        self.lambd = lambd
        self.r = r
        self.tau = tau
        self.tau1 = tau1
        self.tau2 = tau2
        self.a = a
        self.rectangular_kernel = rectangular_kernel
        if num_boundry is None:
            self.boundry = n_clusters*3
        else:
            self.boundry = num_boundry

    def _get_kernel(self):
        # Get the kernel function
        if self.kernel == 'linear':
            f = lambda X, X_: X
        elif self.kernel == 'polynomial':
            def _poly_kernel(X, X_):
                return np.power((self._gamma)*np.matmul(X, X_.T) + self.coef0, self.degree)
            f = _poly_kernel
        elif self.kernel == 'rbf':
            def _rbf_kernel(X, X_):
                n = X.shape[0]
                m = X_.shape[0]
                xx = np.dot(np.sum(np.power(X, 2), 1).reshape(n, 1), np.ones((1, m)))
                zz = np.dot(np.sum(np.power(X_, 2), 1).reshape(m, 1), np.ones((1, n)))
                return np.exp(-self._gamma*(xx + zz.T - 2 * np.dot(X, X_.T)))
            f = _rbf_kernel
        elif self.kernel == 'sigmoid':
            def _sigmoid_kernel(X, X_):
                return np.tanh(self._gamma*np.matmul(X, X_.T) + self.coef0)
            f = _sigmoid_kernel
        return f

    def _S_lambda(self, a, lambda_):
        l = lambda_ / 2
        a_ = l*(a < (-l)) - l*(a > l)
        return np.where(a_ != 0, a + a_, 0)

    def _solve(self, H, G):
        H = np.array(H)
        G = np.array(G)

        v = np.zeros(G.shape[0])
        eta = np.zeros(G.shape[0])
        alpha = np.zeros(G.shape[0])

        z = np.random.randn(H.shape[1]) * 1e-2
        gamma1 = np.random.randn(H.shape[0]) * 1e-2
        gamma2 = np.random.randn(z.shape[0]) * 1e-2

        for _ in range(self.max_iter):
            nu, delta = 1., .99
            S = self.C * np.eye(z.shape[0])
            U = np.diag(np.sign(G @ z))
            A = U @ G
            AtA = A.T @ A
            eyeA = np.eye(A.shape[1])
            S[:, -1] = 0
            S[-1, :] = 0
            zo = z
            for i in range(self.max_iter):
                At_A = np.linalg.inv(eyeA + self.r*nu*(AtA))
                z_ = ( At_A ) @ (z
                                 - nu * (H.T @ gamma1 + S.T @ gamma2)
                                 + (nu * self.lambd / 2) * A.T.sum(-1)
                                 + nu * A.T @ alpha
                                 - nu * self.r * A.T @ eta
                                 + nu * self.r * A.T @ v
                                 + (nu * self.r * A.T).sum(-1)
                                 )

                tetha = 1 / np.sqrt(1 + 2*self.a*delta)
                nu = nu * tetha
                delta = delta / tetha

                zh_ = z_ + tetha*(z_ - z)
                diff = np.linalg.norm(z_ - z)
                tmp1 = (gamma1 + delta * (H @ zh_)) / (1 + delta * self.tau)
                gamma1 = tmp1 / np.maximum(np.abs(tmp1).max(), 1.)

                tmp2 = (gamma2 + delta * (S @ zh_)) / (1 + delta * self.tau)
                gamma2 = tmp2 / np.maximum(np.abs(tmp2).max(), 1.)
                z = z_
                if diff < self.tol_alpha:
                    break

            Gz = G @ z
            Uz = np.diag(np.sign(Gz))
            Az = Uz @ Gz
            for _ in range(self.max_iter):
                eta_ = eta - self.tau1*((self.C -self.lambd - alpha) + self.r*(Az + eta - v - 1))
                eta_ = self._S_lambda(eta_, self.lambd)
                diff = np.linalg.norm(eta_ - eta)
                eta = eta_
                if diff < self.tol_alpha:
                    break

            for _ in range(self.max_iter):
                v_ = v - self.tau2*(alpha + self.r*(v - Az - eta + 1))
                v_ = self._S_lambda(v_, self.lambd)
                diff = np.linalg.norm(v_ - v)
                v = v_
                if diff < self.tol_alpha:
                    break

            alpha = alpha + self.r*(v - Az - eta + 1)

            err = np.linalg.norm(zo - z) / np.linalg.norm(zo)
            if err <= self.tol_beta:
                break
        return z[:-1], z[-1]

    def _get_new_weights(self, X, y, c):
        # Given X and y, create the weights for the hyperplane
        A = X[y == c]
        B = X[y != c]

        mA = self._kernel(A, self.X_)
        mB = self._kernel(B, self.X_)

        H = np.column_stack([mA, np.ones((A.shape[0], 1))])
        G = np.column_stack([mB, np.ones((B.shape[0], 1))])
        w, b = self._solve(H, G)

        return w, b


    def _fit_step(self, X, y):
        # Calculate the new weight vector
        new_weights = []
        new_biases = []
        for i in range(self.n_clusters):
            w, b = self._get_new_weights(X, y, i)
            new_weights.append(w)
            new_biases.append(b)

        # Calculate the difference between the old and new weights
        new_weights = np.stack(new_weights, axis=0)
        new_biases = np.stack(new_biases, axis=0)
        if self.weights is None:
            diff = float('inf')
        else:
            diff = LA.norm(new_weights - self.weights) + LA.norm(new_biases - self.biases)
        self.weights = new_weights
        self.biases = new_biases
        # Update the clusters
        mX = self._kernel(X, self.X_)
        distances = np.abs(mX @ self.weights.T + self.biases[None, :])
        #distances /= np.linalg.norm(self.weights.T, ord=1, axis=0, keepdims=True)
        # Check for empty clusters
        new_y = np.argmin(distances, 1)
        for i in range(self.n_clusters):
            if np.sum(new_y == i) != 0:
                continue
            n = np.argmin(distances[:, i])
            d = distances[n, i]
            new_y[n] = i
            distances[n, :] = float('inf')
            distances[n, i] = d
        # Get the medoids
        medoids = []
        dist_to_closest = distances.min(-1)
        boundry = np.argsort(dist_to_closest)
        boundry = boundry[-self.boundry:]
        for i in range(self.n_clusters):
            c_dists = distances[:, i]
            c_dists[new_y != i] = float('inf')
            m = np.argmin(c_dists)
            if not np.isinf(c_dists[m]):
                medoids.append(int(m))
            else:
                medoids.append(None)
        self._medoids = medoids
        self._boundry = boundry
        return new_y, np.sum(y != new_y)

    def fit(self, X, y=None):
        if self.weights is None:
            self.n_features = X.shape[-1]
            self.weights = None
            self.biases = None
        elif self.weights.shape[-1] != self.n_features:
            raise ValueError(f'Samples must have the same number of features as weights')

        if self.gamma == 'scale':
            self._gamma = 1 / (2*X.shape[-1] * X.var())
        elif self.gamma == 'auto':
            self._gamma = 1 / X.shape[-1]
        elif self.gamma >= 0.:
            self._gamma = self.gamma
        else:
            raise ValueError("Value of gamma {self.gamma} is not supported, must be "+
                             "One of 'scale', 'auto', or a positive number")

        self._kernel = self._get_kernel()

        if y is None:
            if self.init == 'kmeans':
                km = KMeans(n_clusters=self.n_clusters, n_init='auto', init='random').fit(X)
                y = km.labels_
            elif self.init == 'gmm':
                gmm = GMM(n_components=self.n_clusters,init_params='k-means++').fit(X)
                y = gmm.predict(X)
            elif self.init == 'nng':
                y = nng_clustering(X, self.n_clusters, p=1)
            elif self.init == 'random':
                y = np.random.RandomState(42).randint(0, self.n_clusters, size=X.shape[0])
            else:
                ValueError(f'Unknown initialization method "{self.init}"')
        elif y.shape[0] != X.shape[0]:
            raise ValueError(f'Samples must have the same first dimension as labels')
        elif y.max() >= self.n_clusters or y.min() < 0:
            raise ValueError(f'All labels must be between 0 and {self.n_clusters}')

        self._niter = 0

        if self.rectangular_kernel is not None:
            l = int(X.shape[-1] * self.rectangular_kernel)
            self.X_ = X[:l]
        else:
            self.X_ = X
        while True:
            if self._niter > self.max_outer_iter:
                break
            yn, diff = self._fit_step(X, y)
            self._niter += 1
            if diff == 0:
                break
            y = yn

        return y

    def predict(self, X):
        mX = self._kernel(X, self.X_)
        distances = np.abs(mX @ self.weights.T + self.biases[None, :])
        #distances /= np.linalg.norm(self.weights.T, ord=1, axis=0, keepdims=True)
        # Check for empty clusters
        pred = np.argmin(distances, 1)
        return pred

    def get_params(self, deep=True):
        return {
                'C': self.C,
                'n_clusters': self.n_clusters,
                'kernel': self.kernel,
                'lambd': self.lambd,
                'r': self.r,
                'tau': self.tau,
                'a': self.a,
                'gamma': self.gamma,
                'tol_alpha': self.tol_alpha,
                'tol_beta': self.tol_beta,
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

import pandas as pd

def encode_labels(df, target_column):
    return pd.factorize(df[target_column])[0]

def calculate_accuracy(true_labels, predicted_labels):
    true_labels = OneHotEncoder(true_labels)[:, :, np.newaxis]
    predicted_labels = OneHotEncoder(predicted_labels)[:, np.newaxis, :]

    mat = ((true_labels + predicted_labels) > 1)
    confusion = mat.sum(0)
    best = 0
    for _ in range(len(confusion)):
        i, n = np.unravel_index(np.argmax(confusion, axis=None), confusion.shape)
        best += confusion[i, n]
        confusion[i, :] = 0
        confusion[:, n] = 0
    return (best / true_labels.shape[0])

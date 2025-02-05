from cte import N_INSTANCES
import numpy as np
from modAL.utils import multi_argmax
from sklearn.base import BaseEstimator
from modAL.utils.data import modALinput
from modAL.density import information_density
from modAL.uncertainty import classifier_uncertainty
from modAL.utils.combination import make_linear_combination


def classifier_density(classifier: BaseEstimator, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    # return information_density(X, 'euclidean')
    return information_density(X, 'manhattan')


def update_epsilon_values(e):
    return make_linear_combination(classifier_uncertainty, classifier_density, weights=[e, 1 - e])


def custom_query_strategy(classifier, X, epsilon_argument):
    linear_combination = update_epsilon_values(epsilon_argument)

    # measure the utility of each instance in the pool
    utilities = linear_combination(classifier, X)
    # select the indices of the instances to be queried
    query_idx = multi_argmax(utilities, n_instances=N_INSTANCES)
    # return the indices and the instances
    return query_idx, X[query_idx]


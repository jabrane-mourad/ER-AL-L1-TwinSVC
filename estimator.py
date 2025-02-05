from modAL import ActiveLearner

from Extending_modAL import custom_query_strategy
from cte import N_INITIAL, ESTIMATOR


def get_active_learner(strategy, X_train=None, y_train=None):
    if N_INITIAL:
            return ActiveLearner(estimator=ESTIMATOR, X_training=X_train, y_training=y_train, query_strategy=custom_query_strategy)
    return ActiveLearner(estimator=ESTIMATOR, query_strategy=custom_query_strategy)


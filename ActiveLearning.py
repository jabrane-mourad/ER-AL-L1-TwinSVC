import numpy as np
from Score import Score
from cte import N_QUERIES, N_INITIAL
from strategy import get_epsilon_value
from tools import get_mesures


def start_active_learning(learner, strategy, x_pool, y_pool, x_test, y_test):
    score = Score(0, 0, 0, 0)
    scores = []
    if N_INITIAL == 0:
        scores.append(score)
    X = []
    y = np.array([])
    for i in range(N_QUERIES):
        epsilon = get_epsilon_value(strategy, i)
        learner, x_pool, y_pool, Xi, yi = get_query(learner, x_pool, y_pool, epsilon)
        X = np.vstack([X, Xi]) if len(X) > 0 else Xi
        y = np.append(y, yi)
        score = get_mesures(learner, x_test, y_test)
        print(f'strategy == {strategy}   ||   i=={i}/{N_QUERIES}  || {score}')
        scores.append(score)
    return learner, scores


def get_query(learner, x_pool, y_pool, epsilon):
    query_idx, query_inst = learner.query(x_pool, epsilon_argument=epsilon)

    X = x_pool[query_idx]
    y = np.array(y_pool[query_idx], dtype=int)
    learner.teach(X=X, y=y)

    x_pool = np.delete(x_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)
    return learner, x_pool, y_pool, X, y

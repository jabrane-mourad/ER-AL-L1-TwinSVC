from cte import N_QUERIES, Exploitation, Exploration, DPQ
def DPQ_strategy(i):
    if i < 0.1 * N_QUERIES / 100: return 0
    return 1

def get_epsilon_value(strategy, i):
    if strategy == DPQ: return DPQ_strategy(i)
    raise Exception("Sorry, strategy not defined")
from copy import deepcopy

from estimator import get_active_learner
from Result import Result
from save import save_final_results
from cte import NI, STRATEGIES
from data_reading import get_data
from sklearn.metrics import confusion_matrix
from ActiveLearning import start_active_learning
from tools import get_pool_data, add_result, get_average_graph
import warnings
warnings.filterwarnings("ignore")

def get_init_dicts():
    accuracy_dict, precision_dict, recall_dict, f1_dict = {}, {}, {}, {}
    for strategy in STRATEGIES:
        accuracy_dict[strategy] = []
        precision_dict[strategy] = []
        recall_dict[strategy] = []
        f1_dict[strategy] = []
    return accuracy_dict, precision_dict, recall_dict, f1_dict


def my_function(strategy, x_initial_bootstrapping, y_initial_bootstrapping, x_pool, y_pool, x_test, y_test):
    learner = get_active_learner(strategy, x_initial_bootstrapping, y_initial_bootstrapping)
    learner, scores = start_active_learning(learner, strategy, x_pool, y_pool, x_test, y_test)
    return Result(strategy, learner, scores)


def main():
    x_train, y_train, x_test, y_test = get_data()
    accuracy_dict, precision_dict, recall_dict, f1_dict = get_init_dicts()
    for i in range(NI):
        print("ni=", i)
        x_initial_bootstrapping, y_initial_bootstrapping, x_pool, y_pool = get_pool_data(x_train, y_train, x_test,
                                                                                         y_test)
        results = []
        for s in STRATEGIES:
            results.append(
                my_function(s,
                           deepcopy(x_initial_bootstrapping), deepcopy(y_initial_bootstrapping),
                            deepcopy(x_pool), deepcopy(y_pool),
                            deepcopy(x_test), deepcopy(y_test)))

        # save_results(results)
        accuracy_dict, precision_dict, recall_dict, f1_dict = add_result(accuracy_dict, precision_dict,
                                                                         recall_dict, f1_dict, results)
    get_average_graph(f1_dict, "F1")
    get_average_graph(accuracy_dict, "Accuracy")
    get_average_graph(precision_dict, "Precision")
    get_average_graph(recall_dict, "Recall")
    save_final_results(accuracy_dict, precision_dict, recall_dict, f1_dict)


if __name__ == "__main__":
    main()

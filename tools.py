import csv
import os
from datetime import datetime
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from Score import Score
from bootstrap_data import get_bootstrap_data
from cte import N_INITIAL, DATA_BASE, N_QUERIES, directory, train, P_QUERIES
from save import save_average_result
import matplotlib.ticker as mtick


def get_pool_data(x_train, y_train, x_test, y_test):
    result = True
    while result:
        try:
            initial_idx = get_bootstrap_data(x_train, x_test)
            initial_idx=list(filter(None, initial_idx))
            x_initial_bootstrapping, y_initial_bootstrapping = x_train[initial_idx], y_train[initial_idx]
            x_pool, y_pool = np.delete(x_train, initial_idx, axis=0), np.delete(y_train, initial_idx, axis=0)
            if N_INITIAL<3:
                result = 0
            else:
                result = get_diff_class(y_initial_bootstrapping)
        except Exception as e:
            print("Exception:", e)
            pass
    return x_initial_bootstrapping, y_initial_bootstrapping, x_pool, y_pool


def get_diff_class(data):
    if np.unique(data).size == 2:
        return False
    return True


def get_mesures(learner, x_test, y_test):
    y_predicted = learner.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    precision =precision_score(y_test, y_predicted,zero_division=0)
    recall = recall_score(y_test, y_predicted,zero_division=0)
    f1 = f1_score(y_test, y_predicted,zero_division=0)
    score = Score(accuracy, precision, recall, f1)
    return score

def show_graph_result(results_list):
    accuracy_dict, precision_dict, recall_dict, f1_dict = {}, {}, {}, {}
    for result in results_list:
        accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
        for score in result.scores_data:
            accuracy_scores.append(score.accuracy)
            precision_scores.append(score.precision)
            recall_scores.append(score.recall)
            f1_scores.append(score.f1)
        accuracy_dict[result.strategy] = accuracy_scores
        precision_dict[result.strategy] = precision_scores
        recall_dict[result.strategy] = recall_scores
        f1_dict[result.strategy] = f1_scores
    plot_xj(accuracy_dict, "Accuracy")
    plot_xj(precision_dict, "Precision")
    plot_xj(recall_dict, "Recall")
    plot_xj(f1_dict, "F1")


def add_result(accuracy_dict, precision_dict, recall_dict, f1_dict, results):
    for result in results:
        accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
        for score in result.scores_data:
            accuracy_scores.append(score.accuracy)
            precision_scores.append(score.precision)
            recall_scores.append(score.recall)
            f1_scores.append(score.f1)
        accuracy_dict[result.strategy].append(accuracy_scores)
        precision_dict[result.strategy].append(precision_scores)
        recall_dict[result.strategy].append(recall_scores)
        f1_dict[result.strategy].append(f1_scores)
    return accuracy_dict, precision_dict, recall_dict, f1_dict



def get_sv_ML():
    if (DATA_BASE == 'S_DBLP_ACM'):                 return 0.988
    elif (DATA_BASE == 'S_DBLP_GoogleScholar'):     return 0.929
    elif (DATA_BASE == 'S_iTunes_Amazon'):          return 0.923
    elif (DATA_BASE == 'S_Walmart-Amazon'):         return 0.678
    elif (DATA_BASE == 'S_BeerAdvo_RateBeer'):      return 0.875
    elif (DATA_BASE == 'S_Amazon_Google'):          return 0.561
    elif (DATA_BASE == 'S_Fodors_Zagats'):          return 1.0
    elif (DATA_BASE == 'D_DBLP_ACM'):               return 0.933
    elif (DATA_BASE == 'D_DBLP_GoogleScholar'):     return 0.885
    elif (DATA_BASE == 'D_iTunes_Amazon'):          return 0.640
    elif (DATA_BASE == 'D_Walmart_Amazon'):         return 0.452
    elif (DATA_BASE == 'D_wdc_phones'):             return 0.851
    elif (DATA_BASE == 'D_wdc_headphones'):         return 0.966
    elif (DATA_BASE == 'T_abt_buy'):                return 0.513
    elif (DATA_BASE == 'T_Amazon_Google'):          return 0.699
    elif (DATA_BASE == 'T_Company'):                return 0.798

def get_sv_DL():
    if (DATA_BASE == 'S_DBLP_ACM'):                 return 0.992
    elif (DATA_BASE == 'S_DBLP_GoogleScholar'):     return 0.959
    elif (DATA_BASE == 'S_iTunes_Amazon'):          return 0.981
    elif (DATA_BASE == 'S_Walmart-Amazon'):         return 0.867
    elif (DATA_BASE == 'S_BeerAdvo_RateBeer'):      return 0.943
    elif (DATA_BASE == 'S_Amazon_Google'):          return 0.758
    elif (DATA_BASE == 'S_Fodors_Zagats'):          return 1.0
    elif (DATA_BASE == 'D_DBLP_ACM'):               return 0.990
    elif (DATA_BASE == 'D_DBLP_GoogleScholar'):     return 0.957
    elif (DATA_BASE == 'D_iTunes_Amazon'):          return 0.980
    elif (DATA_BASE == 'D_Walmart_Amazon'):         return 0.856
    elif (DATA_BASE == 'D_wdc_phones'):             return 0
    elif (DATA_BASE == 'D_wdc_headphones'):         return 0
    elif (DATA_BASE == 'T_abt_buy'):                return 0.893
    elif (DATA_BASE == 'T_Amazon_Google'):          return 0
    elif (DATA_BASE == 'T_Company'):                return 1.00

def plot_percentile_data_from_dict(data_dict,title):
    """
    Plot data points at specific percentiles from multiple datasets contained in a dictionary,
    with dynamic x-axis range formatted as percentages, using different markers for each dataset.

    Parameters:
    - data_dict (dict): A dictionary with keys as labels and values as arrays of y-values.
    """
    # Define the x-values based on the maximum length of the y-values arrays
    x_values = range(len(next(iter(data_dict.values()))))  # Assumes all arrays are the same length

    # Determine the maximum x-value dynamically
    max_x_value = train

    # Define markers and line styles
    markers = cycle(['o', '^','*', 'v', '<', '>', 'p',  'h', 'D', 's'])
    line_styles = cycle(['-', '--', '-.', ':'])

    # Plotting
    plt.figure(figsize=(12, 6))
    # Loop through each dataset in the dictionary
    dict_percent={}
    for label, y_values, marker, line_style in zip(data_dict.keys(), data_dict.values(), markers, line_styles):
        # Calculate percentile values for x and y
        percentiles = np.arange(0, 101, 1)  # Adjusted for 0%, 10%, ..., 100%
        percentile_x_values = np.percentile(x_values, percentiles)
        indices = [np.abs(x_values - px).argmin() for px in percentile_x_values]
        selected_x_values = [x_values[i] for i in indices]
        selected_y_values = [y_values[i] for i in indices]
        dict_percent["x_values"]=percentiles
        dict_percent[label] = selected_y_values
        # Plot each dataset with a unique marker and line style
        plt.plot(selected_x_values, selected_y_values, label=label)
        # plt.plot(selected_x_values, selected_y_values, f'{line_style}{marker}', label=label)

    # Format the x-axis as percentages of the maximum x-value
    plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f'{(x / max_x_value) * 100:.0f}%'))
    plt.xticks(np.linspace(0, N_QUERIES, 11),
               [f'{(i / max_x_value) * 100:.0f}%' for i in np.linspace(0, N_QUERIES, 11)])
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    if title=="F1":
        plt.axhline(y=get_sv_DL(), color='black', linestyle='solid', label="passive DL")
        plt.axhline(y=get_sv_ML(), color='black', linestyle='--', label="passive ML")

    # plt.title("average_" + title)
    plt.xlabel('Iteration')
    plt.ylabel(title + '-Score')
    plt.legend()
    plt.grid(True)
    # plt.savefig(f'output/{P_QUERIES}/{DATA_BASE}/{directory}/average_{title}.pdf',bbox_inches='tight', format='pdf')
    # plt.show()
    return dict_percent






def get_average_graph(dict, title):
    for k, v in dict.items():
        # Calculate the column-wise average and round to 3 decimal places
        column_avg = np.mean(np.array(v), axis=0)
        dict[k]=list( np.round(column_avg, 3))
    save_average_result(dict, title)
    plot_percentile_data_from_dict(dict, title)

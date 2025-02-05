import csv
import os
import pandas as pd

from cte import DATA_BASE, N_INITIAL, directory, train, P_INITI, P_QUERIES


def save_final_results(accuracy_dict, precision_dict, recall_dict, f1_dict, add_header=True):
    header = ['QUERY', 'Accuracy', 'Precision', 'Recall', 'F1']
    with open(f'output/{P_QUERIES}/{DATA_BASE}/{directory}/average_result.csv', 'w', newline='',encoding='UTF8') as f:
        writer = csv.writer(f)
        if add_header:
            writer.writerow(header)
        for k, v in accuracy_dict.items():
            lenn = len(v) - 1
            writer.writerow(
                [k, accuracy_dict[k][lenn], precision_dict[k][lenn], recall_dict[k][lenn], f1_dict[k][lenn]])
def save_average_result(data,title):
    if not os.path.exists(f'output/{P_QUERIES}/{DATA_BASE}/{directory}/'):
        os.makedirs(f'output/{P_QUERIES}/{DATA_BASE}/{directory}/')

    df = pd.DataFrame(data)
    df.insert(0, 'Iteration', range(0, len(df)))
    df.insert(1, 'Percentage', df['Iteration'] * 100 / train)
    df.to_csv(f'output/{P_QUERIES}/{DATA_BASE}/{directory}/all_iterationts_results_{title}.csv', header=True,index=False)


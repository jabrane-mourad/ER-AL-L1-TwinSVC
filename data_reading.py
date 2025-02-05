import numpy as np
import pandas as pd

from cte import DATA_BASE

encoding_list = ['utf_8', 'ascii', 'big5', 'big5hkscs', 'cp037', 'cp273', 'cp424', 'cp437', 'cp500', 'cp720', 'cp737'
    , 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857', 'cp858', 'cp860', 'cp861', 'cp862'
    , 'cp863', 'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949', 'cp950'
    , 'cp1006', 'cp1026', 'cp1125', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254'
    , 'cp1255', 'cp1256', 'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr'
    , 'gb2312', 'gbk', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2'
    , 'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1', 'iso8859_2'
    , 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7', 'iso8859_8', 'iso8859_9'
    , 'iso8859_10', 'iso8859_11', 'iso8859_13', 'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab'
    , 'koi8_r', 'koi8_t', 'koi8_u', 'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2'
    , 'mac_roman', 'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213', 'utf_32'
    , 'utf_32_be', 'utf_32_le', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_7', 'utf_8_sig']


def read_data(path):
    for encoding in encoding_list:
        worked = True
        try:
            df = pd.read_csv(path, encoding=encoding)
        except:
            worked = False
        if worked:
            print(encoding, ':\n', df.head())
            return df;


def get_data():
    print(f'\n \n ________DATABASE={DATA_BASE}________\n \n')

    dataset_train = pd.read_csv(f'datasets/{DATA_BASE}/features_train.csv')
    dataset_test = pd.read_csv(f'datasets/{DATA_BASE}/features_test.csv')
    print("dataset_train shape:", dataset_train.shape)
    print("dataset_test shape:", dataset_test.shape)
    X_train = dataset_train.iloc[:, 4:]
    median_vals = X_train.median()
    X_train = X_train.fillna(median_vals).values
    y_train = dataset_train.iloc[:, 3].values.astype(np.int8)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.] = 1.
    X_train = (X_train - mean) / std

    X_test = dataset_test.iloc[:, 4:]
    X_test = X_test.fillna(median_vals).values
    X_test = (X_test - mean) / std
    y_test = dataset_test.iloc[:, 3].values.astype(np.int8)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    return X_train, y_train, X_test, y_test
import csv

import nltk

from CTE import DATA_BASE_DIRECTORY, FEATURES_TRAIN_DIRECTORY, FEATURES_TEST_DIRECTORY, DATA_BASE, \
    FEATURES_VALID_DIRECTORY
from get_data import read_data
from similarityutils import *
from gensim.models import Word2Vec, KeyedVectors
from displayutils import *
import re
from dateutil.parser import parse
from datetime import datetime
import pytz
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def readData(directory):
    # source = read_data(f'{DATA_BASE_DIRECTORY}/tableA.csv')
    source = read_data('/home/lipim/Desktop/code_doc/createFV/datasets/T_abt_buy/tableA.csv')
    print("Source file records:", len(source))
    # target = read_data(f'{DATA_BASE_DIRECTORY}/tableB.csv')
    target = read_data('/home/lipim/Desktop/code_doc/createFV/datasets/T_abt_buy/tableB.csv')
    print("Target file records:", len(target))

    gold_standard_train = read_data('/home/lipim/Desktop/code_doc/createFV/datasets/T_abt_buy/gold_standard_train.csv')
    print("Correspondences in the gold_standard_train:", len(gold_standard_train))

    gold_standard_test = read_data('/home/lipim/Desktop/code_doc/createFV/datasets/T_abt_buy/gold_standard_test.csv')
    print("Correspondences in the gold_standard_test:", len(gold_standard_test))
    gold_standard_valid = read_data(f'/home/lipim/Desktop/code_doc/createFV/datasets/T_abt_buy/gold_standard_valid.csv')
    print("Correspondences in the gold_standard_test:", len(gold_standard_test))

    return source, target, gold_standard_train, gold_standard_test, gold_standard_valid


def getTypesofData(data):
    dict_types = dict()
    # return dictionary
    for column in data:
        column_values = data[column].dropna()
        type_list = list(set(column_values.map(type).tolist()))

        if len(type_list) == 0:
            "No type could be detected. Default (string) will be assigned."
            dict_types[column] = 'str'
        elif len(type_list) > 1:
            "More than one types could be detected. Default (string) will be assigned."
            dict_types[column] = 'str'
        else:
            if str in type_list:
                types_of_column = []
                length = 0
                for value in column_values:
                    length = length + len(value.split())
                    if re.match(r'.?\d{2,4}[-\.\\]\d{2}[-\.\\]\d{2,4}.?', value):
                        types_of_column.append('date')
                avg_length = length / len(column_values)

                if (avg_length > 6): types_of_column.append('long_str')

                if len(set(types_of_column)) == 1:
                    if ('date' in types_of_column):
                        dict_types[column] = 'date'
                    elif ('long_str' in types_of_column):
                        dict_types[column] = 'long_str'
                    else:
                        dict_types[column] = 'str'
                else:
                    "More than one types could be detected. Default (string) will be assigned."
                    dict_types[column] = 'str'
            else:  # else it must be numeric
                dict_types[column] = 'numeric'
    return dict_types


def is_date(string, fuzzy=True):
    try:
        parse(string, fuzzy=fuzzy, default=datetime(1, 1, 1, tzinfo=pytz.UTC))
        return True

    except ValueError:
        return False


def createFeatureVectorFile(source, target, pool, featureFile, keyfeature='id', embeddings=True):
    source_headers = source.columns.values
    target_headers = target.columns.values

    print("Get types of data")
    dict_types_source = getTypesofData(source)
    print(dict_types_source)
    dict_types_target = getTypesofData(target)
    print(dict_types_target)

    common_elements = list(set(source_headers) & set(target_headers) - {keyfeature})
    common_elements_types = dict()
    for common_element in common_elements:
        if (dict_types_source[common_element] is dict_types_target[common_element]):
            common_elements_types[common_element] = dict_types_source[common_element]
        else:
            if (dict_types_source[common_element] == 'long_str' or dict_types_target[common_element] == 'long_str'):
                print(
                    "Different data types in source and target for element %s. Will assign long string" % common_element)
                common_elements_types[common_element] = 'long_str'
            else:
                print("Different data types in source and target for element %s. Will assign string" % common_element)
                common_elements_types[common_element] = 'str'

    # calculate tfidf vectors
    print("Calculate tfidf scores")
    records = dict()
    records["data"] = np.concatenate((source[common_elements].values, target[common_elements].values), axis=0)
    records["ids"] = np.concatenate((source[keyfeature], target[keyfeature]), axis=0)

    tfidfvector_ids = calculateTFIDF(records)

    print("Create similarity based features from", len(common_elements), "common elements")

    similarity_metrics = {
        'str': ['lev', 'jaccard', 'relaxed_jaccard', 'overlap', 'cosine', 'containment'],
        'numeric': ['abs_diff', 'num_equal'],
        'date': ['day_diff', 'month_diff', 'year_diff'],
        'long_str': ['cosine', 'lev', 'jaccard', 'relaxed_jaccard', 'overlap', 'cosine_tfidf', 'containment']
    }

    if not embeddings:
        similarity_metrics['str'].remove('cosine')
        similarity_metrics['long_str'].remove('cosine')

    features = []

    # fix headers
    header_row = []
    header_row.append('source_id')
    header_row.append('target_id')
    header_row.append('pair_id')
    header_row.append('label')
    header_row.append('cosine_tfidf')
    for f in common_elements:
        for sim_metric in similarity_metrics[common_elements_types[f]]:
            header_row.append(f + "_" + sim_metric)

    features.append(header_row)
    word2vec = None
    if embeddings:
        print("Load pre-trained word2vec embeddings")
        filename = 'GoogleNews-vectors-negative300.bin'
        word2vec = KeyedVectors.load_word2vec_format(filename, binary=True)
        print("Pre-trained embeddings loaded")

    tfidfvector_perlongfeature = dict()
    if 'long_str' in common_elements_types.values():
        for feature in common_elements_types:
            if common_elements_types[feature] == 'long_str':
                records_feature = dict()
                records_feature["data"] = np.concatenate((source[feature].values, target[feature].values), axis=0)
                records_feature["ids"] = np.concatenate((source[keyfeature], target[keyfeature]), axis=0)
                tfidfvector_feature = calculateTFIDF(records_feature)
                tfidfvector_perlongfeature[feature] = tfidfvector_feature

    print_progress(0, len(pool), prefix='Create Features:', suffix='Complete')
    ps = PorterStemmer()
    for i in range(len(pool)):
        print_progress(i + 1, len(pool), prefix='Create Features:', suffix='Complete')
        features_row = []
        # metadata
        r_source_id = pool['source_id'].loc[i]
        r_target_id = pool['target_id'].loc[i]

        features_row.append(r_source_id)
        features_row.append(r_target_id)
        features_row.append(str(r_source_id) + "-" + str(r_target_id))
        features_row.append(pool['matching'].loc[i])

        features_row.append(get_cosine_tfidf(tfidfvector_ids, r_source_id, r_target_id))

        for f in common_elements:
            fvalue_source = str(source.loc[source[keyfeature] == r_source_id][f].values[0])
            fvalue_target = str(target.loc[target[keyfeature] == r_target_id][f].values[0])

            if common_elements_types[f] == 'str' or common_elements_types[f] == 'long_str':
                fvalue_source = re.sub('[^A-Za-z0-9]+', ' ', str(fvalue_source.lower())).strip()
                fvalue_target = re.sub('[^A-Za-z0-9]+', ' ', str(fvalue_target.lower())).strip()
            ## if long str remove stopwords and stem
            if common_elements_types[f] == 'long_str':
                cachedStopWords = stopwords.words("english")
                fvalue_source = ' '.join([word for word in fvalue_source.split() if word not in cachedStopWords])
                fvalue_target = ' '.join([word for word in fvalue_target.split() if word not in cachedStopWords])
                # stem
                fvalue_source = ' '.join([ps.stem(word) for word in fvalue_source.split()])
                fvalue_target = ' '.join([ps.stem(word) for word in fvalue_target.split()])

            if f in tfidfvector_perlongfeature:
                typeSpecificSimilarities(common_elements_types[f], fvalue_source, fvalue_target, similarity_metrics,
                                         features_row, word2vec, tfidfvector_perlongfeature[f], r_source_id,
                                         r_target_id)
            else:
                typeSpecificSimilarities(common_elements_types[f], fvalue_source, fvalue_target, similarity_metrics,
                                         features_row, word2vec)

        features.append(features_row)

    print('Created', len(features[0]), 'features for', len(features), 'entity pairs')

    with open(featureFile, mode='w') as feature_file:
        writer = csv.writer(feature_file)
        writer.writerows(features)

    print("Feature file created")


def typeSpecificSimilarities(data_type, valuea, valueb, type_sim_map, features_row, word2vec, tfidfvector=None,
                             r_source_id=None, r_target_id=None):
    # similarity-based features
    values_sim = []
    for sim_metric in type_sim_map[data_type]:
        if valuea == 'nan' or valueb == 'nan' or valuea == '' or valueb == '':
            features_row.append(-1.0)
            values_sim.append(-1)
        elif sim_metric == 'lev':
            features_row.append(get_levenshtein_sim(valuea, valueb))
            values_sim.append(get_levenshtein_sim(valuea, valueb))
        elif sim_metric == 'jaccard':
            features_row.append(get_jaccard_sim(valuea, valueb))
            values_sim.append(get_jaccard_sim(valuea, valueb))
        elif sim_metric == 'relaxed_jaccard':
            features_row.append(get_relaxed_jaccard_sim(valuea, valueb))
            values_sim.append(get_relaxed_jaccard_sim(valuea, valueb))
        elif sim_metric == 'overlap':
            features_row.append(get_overlap_sim(valuea, valueb))
            values_sim.append(get_overlap_sim(valuea, valueb))
        elif sim_metric == 'containment':
            features_row.append(get_containment_sim(valuea, valueb))
            values_sim.append(get_containment_sim(valuea, valueb))
        elif sim_metric == 'cosine':
            features_row.append(get_cosine_word2vec(valuea, valueb, word2vec))
            values_sim.append(get_cosine_word2vec(valuea, valueb, word2vec))
        elif sim_metric == 'cosine_tfidf':
            features_row.append(get_cosine_tfidf(tfidfvector, r_source_id, r_target_id))
            values_sim.append(get_cosine_tfidf(tfidfvector, r_source_id, r_target_id))
        elif sim_metric == 'abs_diff':
            features_row.append(get_abs_diff(valuea, valueb))
            values_sim.append(get_abs_diff(valuea, valueb))
        elif sim_metric == 'num_equal':
            features_row.append(get_num_equal(valuea, valueb))
            values_sim.append(get_num_equal(valuea, valueb))
        elif sim_metric == 'day_diff':
            features_row.append(get_day_diff(valuea, valueb))
            values_sim.append(get_day_diff(valuea, valueb))
        elif sim_metric == 'month_diff':
            features_row.append(get_month_diff(valuea, valueb))
            values_sim.append(get_month_diff(valuea, valueb))
        elif sim_metric == 'year_diff':
            features_row.append(get_year_diff(valuea, valueb))
            values_sim.append(get_year_diff(valuea, valueb))
        else:
            print("Unknown similarity metric %s" % sim_metric)
        if (-1 in values_sim and len(set(values_sim)) > 1):
            import pdb
            pdb.set_trace()


if __name__ == '__main__':
    # nltk.download('stopwords')
    source, target, gold_standard_train, gold_standard_test, gold_standard_valid = readData(DATA_BASE_DIRECTORY)

    # createFeatureVectorFile(source, target, gold_standard_train, FEATURES_TRAIN_DIRECTORY)
    createFeatureVectorFile(source, target, gold_standard_test, FEATURES_TEST_DIRECTORY)
    # createFeatureVectorFile(source, target, gold_standard_valid, FEATURES_VALID_DIRECTORY)

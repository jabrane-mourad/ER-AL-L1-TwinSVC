from __future__ import division
# import similarity.levenshtein
from Levenshtein import distance as Levenshtein

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
from dateutil.parser import parse
from displayutils import *
import re




def get_date_type(date_str):
    separator = ''
    if '.' in date_str:
        separator = '.'
    elif '\\' in date_str:
        separator = '\\'
    elif '/' in date_str:
        separator = '/'
    elif '-' in date_str:
        separator = '-'
    else:
        return None
    try:
        date_parts = [d.strip() for d in date_str.split(separator)]
        if re.match('\\d{4}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{1,2}', date_str):
            return datetime.datetime.strptime(date_str, '%Y' + separator + '%m' + separator + '%d').date()
        if re.match('\\d{1,2}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{4}', date_str):
            return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%Y').date()
        if re.match('\\d{2}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{1,2}', date_str):
            p = re.compile('\\d+')
            splitted_date = p.findall(date_str)
            if int(splitted_date[0]) < 32 and int(splitted_date[1]) < 13:
                return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%y').date()
            if int(splitted_date[0]) > 32:
                return datetime.datetime.strptime(date_str, '%y' + separator + '%m' + separator + '%d').date()
            try:
                return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%y').date()
            except:
                try:
                    return datetime.datetime.strptime(date_str, '%y' + separator + '%m' + separator + '%d').date()
                except:
                    print('Unknown pattern or invalid date: %s' % date_str)
                    return None

        else:
            return parse(date_str, fuzzy=True)
    except:
        f = open('unparseddates.txt', 'a')
        f.write(date_str + '\n')
        f.close()
        return None


def get_day_diff(date1, date2):
    if date1 == 'nan' or date2 == 'nan':
        return -1.0
    date1_ = get_date_type(date1)
    date2_ = get_date_type(date2)
    if date1_ == None or date2_ == None:
        return -1.0
    delta = date1_.day - date2_.day
    return abs(delta)


def get_month_diff(date1, date2):
    if date1 == 'nan' or date2 == 'nan':
        return -1.0
    date1_ = get_date_type(date1)
    date2_ = get_date_type(date2)
    if date1_ == None or date2_ == None:
        return -1.0
    delta = date1_.month - date2_.month
    return abs(delta)


def get_year_diff(date1, date2):
    if date1 == 'nan' or date2 == 'nan':
        return -1.0
    date1_ = get_date_type(date1)
    date2_ = get_date_type(date2)
    if date1_ == None or date2_ == None:
        return -1.0
    difference = abs(date1_.year - date2_.year)
    if len(date1) != len(date2) and difference % 100 == 0:
        difference = 0
    return difference


def get_num_equal(num1, num2):
    if num1 == 'nan' or num2 == 'nan':
        return -1.0
    try:
        num1_ = float(num1)
        num2_ = float(num2)
        if num1_ == num2_:
            return 1.0
        return 0.0
    except:
        return -1


def get_abs_diff(num1, num2):
    if num1 == 'nan' or num2 == 'nan':
        return -1.0
    try:
        num1_ = float(num1)
        num2_ = float(num2)
        return abs(num1_ - num2_)
    except:
        return -1


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    else:
        return float(len(c)) / float(len(a) + len(b) - len(c))


def get_relaxed_jaccard_sim(str1, str2):
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    a = set(str1.split())
    b = set(str2.split())
    c = []
    for a_ in a:
        for b_ in b:
            if get_levenshtein_sim(a_, b_) > 0.7:
                c.append(a_)

    intersection = len(c)
    min_length = min(len(a), len(b))
    if intersection > min_length:
        intersection = min_length
    return float(intersection) / float(len(a) + len(b) - intersection)


def get_containment_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    elif len(a) == 0 or len(b) == 0:
        return -1.0
    else:
        return float(len(c)) / float(min(len(a), len(b)))


def get_levenshtein_sim(str1, str2):
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    else:
        max_length = max(len(str1), len(str2))
        return 1.0 - Levenshtein(str1, str2) / max_length


def get_missing(str1, str2):
    if str1 == 'nan' or str2 == 'nan':
        return 1.0
    else:
        return 0.0


def get_overlap_sim(str1, str2):
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    elif str1 == str2:
        return 1.0
    else:
        return 0.0


def get_cosine_word2vec(str1, str2, model):
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    # elif str1.replace(' ', '') in model.key_to_index and str2.replace(' ', '') in model.key_to_index:
    #     idx1 = model.key_to_index[str1.replace(' ', '')]
    #     idx2 = model.key_to_index[str2.replace(' ', '')]
    #     return model.similarity_by_index(idx1, idx2)
    str11 = str1.replace(' ', '')
    str22 = str2.replace(' ', '')

    if str11 in model.key_to_index and str22 in model.key_to_index:
        vec1 = model.get_vector(str11)
        vec2 = model.get_vector(str22)
        similarity_score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return similarity_score

    else:
        return 0.0


def get_cosine_tfidf(tfidf_scores_ids, sourceID, targetID):
    source_index = np.where(tfidf_scores_ids['ids'] == sourceID)
    target_index = np.where(tfidf_scores_ids['ids'] == targetID)
    score = cosine_similarity(np.asarray(tfidf_scores_ids['scores'][source_index].todense()),
                              np.asarray(tfidf_scores_ids['scores'][target_index].todense()))
    return score[0][0]


def calculateTFIDF(records):
    records_data = records['data']
    concat_records = []
    for row in records_data:
        if (isinstance(row, np.ndarray)):  # tfidf based on  more that one features
            concat_row = ''
            for value in row:
                if not pd.isnull(value):
                    if type(value) is str:
                        if value.lower() != 'nan':
                            value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(value))
                            concat_row += ' ' + value
                    else:  # tfidf based on one feature
                        value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(value))
                        concat_row += ' ' + str(value)

            concat_records.append(concat_row)
        else:
            if pd.isnull(row):
                concat_records.append("")
            else:
                value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(row.lower()))
                concat_records.append(value)

    tf_idfscores = TfidfVectorizer(encoding='latin-1').fit_transform(concat_records)
    tf_idf = dict()
    tf_idf['ids'] = records['ids']
    tf_idf['scores'] = tf_idfscores

    return tf_idf

import pandas as pd
import numpy as np
from scipy.stats import skew, entropy
from sklearn.preprocessing import MinMaxScaler
import pickle

import csv

def get_absolute_change_feature(row):
    mean_row = 0
    for i in range(0, len(row) - 1):
        mean_row = mean_row + np.abs(row[(i + 1)] - row[i])
    if len(row) == 0:
        return 0
    return mean_row / len(row)


def get_skew_feature(row):
    return skew(row)


def get_co_relation_feature(row):
    correlation=np.correlate(np.array(row), np.array(row))
    return float(correlation)


def get_mean_feature(row):
    diff=np.diff(row)
    absolute_val=np.abs(diff)
    return np.mean(absolute_val)


def get_entropy(row):
    return entropy(row)


def get_rms_feature(row):
    rms_feature = 0
    for index in range(0, len(row) - 1):
        rms_feature = rms_feature + np.square(row[index])
    if len(row) == 0:
        return 0
    return np.sqrt(rms_feature / len(row))


def get_polynomial_fit_feature(row):
    inter=np.linspace(0, len(row) - 1, len(row))
    polyfit_feature = np.polyfit(inter, row, 1)
    return polyfit_feature[0]


def get_min_feature(row):
    return min(row)


def get_max_feature(row):
    return max(row)


def get_features_matrix(data_frame):
    # pickle_compat.patch()
    feature_df = pd.DataFrame()
    for _, row_df in data_frame.iterrows():
        # print(row)
        row = row_df.tolist()
        feature_row = {
            'row_min_feature': get_min_feature(row),
            'row_max_feature': get_max_feature(row),
            'row_rms_feature': get_rms_feature(row),
            'row_mac_feature': get_absolute_change_feature(row),
            'row_skew_matrix_feature': get_skew_feature(row),
            'row_co-relation_feature': get_co_relation_feature(row),
            'row_mean_feature': get_mean_feature(row),
            'row_entropy_feature': get_entropy(row),
            'row_polynomial_fit_feature': get_polynomial_fit_feature(row)
        }
        feature_df = feature_df.append(feature_row, ignore_index=True)
    return feature_df


def predict():
    res = []
    print('testing_data: %s' % testing_data)
    for row in testing_data.values:
        row_list = list(row)
        idx = row_list.index(max(row_list))
        res.append(1 if 1 <= idx <= 10 else 0)
    return res


def write_result():
    with open('Result.csv', 'w') as result_file:
        writer = csv.writer(result_file)
        for p in predict_data:
            writer.writerow([p])


if __name__ == '__main__':
    testing_data = pd.read_csv("test.csv", header=None)

    predict_data = predict()
    write_result()

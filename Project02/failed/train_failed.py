import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, entropy
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import classification_report
import pickle

meal_and_insulin_intake = pd.read_csv(
    "InsulinData.csv",
    usecols=['BWZ Carb Input (grams)', 'Date', 'Time'],
    parse_dates=[['Date', 'Time']]
)

meal_and_insulin_intake.dropna(subset=['BWZ Carb Input (grams)'], inplace=True)
meal_and_insulin_intake = meal_and_insulin_intake[meal_and_insulin_intake['BWZ Carb Input (grams)'] > 0]

meal_and_insulin_intake.sort_values(by=['Date_Time'], ascending=True, inplace=True)
meal_and_insulin_intake['flag'] = True
for insulin_index in range(1, meal_and_insulin_intake.shape[0]):
    diff = (meal_and_insulin_intake.iloc[insulin_index, 0] -
            meal_and_insulin_intake.iloc[insulin_index - 1, 0]).seconds / 60
    if diff < 30:
        meal_and_insulin_intake.iloc[insulin_index - 1, 2] = False
        meal_and_insulin_intake.iloc[insulin_index, 2] = False

# print(meal_and_insulin_intake)

meal_and_insulin_intake = meal_and_insulin_intake[meal_and_insulin_intake['flag'] == True]

for i in range(1, meal_and_insulin_intake.shape[0]):
    diff = (meal_and_insulin_intake.iloc[i, 0] - meal_and_insulin_intake.iloc[i - 1, 0]).seconds / 60
    if diff < 120:
        meal_and_insulin_intake.iloc[i - 1, 2] = False

meal_and_insulin_intake = meal_and_insulin_intake[meal_and_insulin_intake['flag'] == True]


def execute_folds_on_features(X, Y):
    kfold_algo = KFold(5, shuffle=True, random_state=1)
    for index_train, index_test in kfold_algo.split(X, Y):
        train_x, test_x = X.iloc[index_train], X.iloc[index_test]
        train_y, test_y = Y.iloc[index_train], Y.iloc[index_test]
        model = GaussianProcessClassifier(kernel=(1.0 * RBF(1)), random_state=0)
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)


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
    correlation = np.correlate(np.array(row), np.array(row))
    return float(correlation)


def get_mean_feature(row):
    diff = np.diff(row)
    absolute_val = np.abs(diff)
    return np.mean(absolute_val)


def get_entropy(row):
    return entropy(row)


def get_rms_feature(row):
    rms_feature = 0
    for i in range(0, len(row) - 1):
        rms_feature = rms_feature + np.square(row[i])
    if len(row) == 0:
        return 0
    return np.sqrt(rms_feature / len(row))


def get_polynomial_fit_feature(row):
    inter = np.linspace(0, len(row) - 1, len(row))
    polyfit_feature = np.polyfit(inter, row, 1)
    return polyfit_feature[0]


def get_min_feature(row):
    return min(row)


def get_max_feature(row):
    return max(row)


def get_features_matrix(data_frame):
    feature_df = pd.DataFrame()
    for row in data_frame:
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
        feature_df = feature_df._append(feature_row, ignore_index=True)
    return feature_df


initial_file_data_cgm = pd.read_csv(
    filepath_or_buffer="CGMData.csv",
    usecols=['Sensor Glucose (mg/dL)', 'Date', 'Time'],
    parse_dates=[['Date', 'Time']]
)

initial_file_data_cgm.dropna(subset=['Sensor Glucose (mg/dL)'], inplace=True)

print('#0')

meal_data_list = []
for i in range(0, meal_and_insulin_intake.shape[0]):
    meal_time = meal_and_insulin_intake.iloc[i]['Date_Time']
    meal_start_time = meal_time - pd.Timedelta(minutes=30)
    meal_end_time = meal_time + pd.Timedelta(hours=2)
    valid_cgm_df = initial_file_data_cgm[
        (initial_file_data_cgm['Date_Time'] >= meal_start_time) & (initial_file_data_cgm['Date_Time'] <= meal_end_time)]
    meal_data = valid_cgm_df['Sensor Glucose (mg/dL)'].tolist()
    if len(meal_data) == 30:
        meal_data_list.append(meal_data)

no_meal_data_list = []
for i in range(0, meal_and_insulin_intake.shape[0]):
    meal_time = meal_and_insulin_intake.iloc[i]['Date_Time']
    meal_end_time = meal_time + pd.Timedelta(hours=2)
    meal_end_2_hr = meal_time + pd.Timedelta(hours=4)
    valid_cgm_df = initial_file_data_cgm[
        (initial_file_data_cgm['Date_Time'] >= meal_end_time) & (initial_file_data_cgm['Date_Time'] <= meal_end_2_hr)]
    no_meal_data = valid_cgm_df['Sensor Glucose (mg/dL)'].tolist()
    if len(no_meal_data) == 24:
        no_meal_data_list.append(no_meal_data)

X = get_features_matrix(meal_data_list)
Y = get_features_matrix(no_meal_data_list)

X['class'] = 1
Y['class'] = 0

# print('#1')

training_dataset_df = pd.concat([X, Y])

execute_folds_on_features(training_dataset_df.iloc[:, :-1], training_dataset_df.iloc[:, -1])

# print('after execute_folds_on_features')

classes = training_dataset_df['class'].tolist()
training_feature_data_df = training_dataset_df.drop(['class'], axis=1)
training_data = training_feature_data_df.values.tolist()
training_data, classes = shuffle(training_data, classes)
x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(training_data, classes, train_size=0.8)

# print('#2')

normalizer = MinMaxScaler()

# print('x_train_data:')
# print(x_train_data)

x_train_data = normalizer.fit_transform(x_train_data)
x_test_data = normalizer.transform(x_test_data)

GPC_model = GaussianProcessClassifier(kernel=(1.0 * RBF(1)), random_state=0)
GPC_model.fit(x_train_data, y_train_data)

# print('#3')

pickle.dump(GPC_model, open('model.pkl', 'wb'))
pred_y = GPC_model.predict(x_test_data)

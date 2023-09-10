import csv
import numpy as np
import pandas
from datetime import datetime
from datetime import time

CGM_FILENAME = 'CGMData.csv'
CGM_COLUMNS = ['Date', 'Time', 'Sensor Glucose (mg/dL)']

INSULIN_FILENAME = 'InsulinData.csv'
INSULIN_COLUMNS = ['Date', 'Time', 'BWZ Carb Input (grams)']


def get_datetime_from_str(date_str: str, time_str: str) -> datetime:
    month, day, year = date_str.split("/")
    hour, minute, second = time_str.split(":")
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))


def get_meal_data():
    for rows in cgm_arr:
        dt = get_datetime_from_str(rows[0], rows[1])
        


if __name__ == '__main__':
    cgm_raw_data = pandas.read_csv(CGM_FILENAME, usecols=CGM_COLUMNS)
    cgm_arr = np.array(cgm_raw_data)

    insulin_raw_data = pandas.read_csv(INSULIN_FILENAME, usecols=INSULIN_COLUMNS)
    insulin_arr = np.array(insulin_raw_data)

    print('cgm_arr: \n%s' % cgm_arr)
    print('insulin_arr: \n%s' % insulin_arr)

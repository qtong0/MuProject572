import csv
import numpy as np
import pandas
from datetime import datetime
from datetime import timedelta
from datetime import date

CGM_FILENAME = 'CGMData.csv'
CGM_COLUMNS = ['Date', 'Time', 'Sensor Glucose (mg/dL)']

INSULIN_FILENAME = 'InsulinData.csv'
INSULIN_COLUMNS = ['Date', 'Time', 'BWZ Carb Input (grams)']


def get_datetime_from_str(date_str: str, time_str: str) -> datetime:
    month, day, year = date_str.split("/")
    hour, minute, second = time_str.split(":")
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))


def get_date_from_str(date_str: str) -> date:
    month, day, year = date_str.split("/")
    return date(int(year), int(month), int(day))


def get_meal_times() -> dict:
    times = {}
    next_meal_dt = None
    for idx, row in enumerate(insulin_arr):
        if not np.isnan(row[2]):
            dt = get_datetime_from_str(row[0], row[1])
            d = get_date_from_str(row[0])
            if next_meal_dt is None or next_meal_dt - timedelta(hours=2) > dt:
                times[d] = times.get(d, []) + [dt]
            next_meal_dt = dt
    return times


def get_meal_data() -> dict:
    data = {}
    for i, row in enumerate(cgm_arr[::-1]):
        row_time = get_datetime_from_str(row[0], row[1])
        row_date = get_date_from_str(row[0])
        for meal_dt in meal_times.get(row_date, []):
            if meal_dt - timedelta(minutes=30) <= row_time < meal_dt + timedelta(hours=2):
                data[meal_dt] = data.get(meal_dt, []) + [row[2]]
    return data


def get_nomeal_data():
    data = []
    return data


if __name__ == '__main__':
    cgm_raw_data = pandas.read_csv(CGM_FILENAME, usecols=CGM_COLUMNS)
    cgm_arr = np.array(cgm_raw_data)

    insulin_raw_data = pandas.read_csv(INSULIN_FILENAME, usecols=INSULIN_COLUMNS)
    insulin_arr = np.array(insulin_raw_data)

    meal_times = get_meal_times()
    print('len(meal_times): %s' % len(meal_times))

    meal_data = get_meal_data()
    # Filter out meal_time where there is less than 30 cgm data
    meal_data = {k: v for k, v in meal_data.items() if len(v) == 30}
    print('len(meal_data): %s' % len(meal_data))

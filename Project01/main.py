import csv

import numpy as np
import pandas
from datetime import datetime
from datetime import time

CGM_FILENAME = 'CGMData.csv'
INSULIN_FILENAME = 'InsulinData.csv'
RESULT_FILENAME = 'Result.csv'

AUTO_MODE_ACTIVE_STR = 'AUTO MODE ACTIVE PLGM OFF'

AUTO_DAYTIME_MAP = {}
AUTO_OVERNIGHT_MAP = {}
AUTO_WHOLEDAY_MAP = {}

MANUAL_DAYTIME_MAP = {}
MANUAL_OVERNIGHT_MAP = {}
MANUAL_WHOLEDAY_MAP = {}

DAYTIME_COUNT = 216
OVERNIGHT_COUNT = 72
WHOLE_DAY_COUNT = 288

# Auto percentage data in the following order:
# Overnight, Daytime, Wholeday
# >180, >250, 70<=p<=180, 70<=p<=150, <70, <54
AUTO_DATA = [[[] for _ in range(6)] for _ in range(3)]

# Manual percentage data in the following order:
# Overnight, Daytime, Wholeday
# >180, >250, 70<=p<=180, 70<=p<=150, <70, <54
MANUAL_DATA = [[[] for _ in range(6)] for _ in range(3)]


def get_datetime_from_str(date_str: str, time_str: str) -> datetime:
    month, day, year = date_str.split("/")
    hour, minute, second = time_str.split(":")
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))


def get_last_auto_mode_date_time() -> datetime:
    insulin_raw_data = pandas.read_csv(INSULIN_FILENAME, usecols=['Date', 'Time', 'Alarm'])

    insulin_arr = np.array(insulin_raw_data)
    last_auto_mode = insulin_arr[insulin_arr[:, 2] == AUTO_MODE_ACTIVE_STR][-1]

    return get_datetime_from_str(
        str(last_auto_mode[0]),
        str(last_auto_mode[1])
    )


def add_to_maps(
        row_date: datetime.day,
        row_time: time,
        val: float,
        is_auto: bool
):
    overnight_map = AUTO_OVERNIGHT_MAP if is_auto else MANUAL_OVERNIGHT_MAP
    daytime_map = AUTO_DAYTIME_MAP if is_auto else MANUAL_DAYTIME_MAP
    wholeday_map = AUTO_WHOLEDAY_MAP if is_auto else MANUAL_WHOLEDAY_MAP
    if row_time < time(6, 0, 0):
        # overnight
        overnight_map[row_date] = overnight_map.get(row_date, []) + [val]
    elif time(6, 0, 0) <= row_time:
        # daytime
        daytime_map[row_date] = daytime_map.get(row_date, []) + [val]
    # whole date
    wholeday_map[row_date] = wholeday_map.get(row_date, []) + [val]


def collect_data():
    for row in cgm_arr:
        dt = get_datetime_from_str(str(row[0]), str(row[1]))
        row_date = dt.date()
        row_time = dt.time()

        if dt >= last_auto_datetime:
            # Auto mode
            if not np.isnan(row[2]):
                add_to_maps(row_date, row_time, row[2], is_auto=True)

        else:
            # Manual mode
            if not np.isnan(row[2]):
                add_to_maps(row_date, row_time, row[2], is_auto=False)


def __process_data(is_auto: bool):
    base_nums = [216, 72, 288]
    data = AUTO_DATA if is_auto else MANUAL_DATA
    maps = [AUTO_OVERNIGHT_MAP, AUTO_DAYTIME_MAP, AUTO_WHOLEDAY_MAP] if is_auto \
        else [MANUAL_OVERNIGHT_MAP, MANUAL_DAYTIME_MAP, MANUAL_WHOLEDAY_MAP]
    for idx, auto_map in enumerate(maps):
        base_num = base_nums[idx]
        for dt, nums in auto_map.items():
            # Ignore data when there is missing
            if len(nums) == base_num:
                data[idx][0].append(sum(f > 180 for f in nums) / float(WHOLE_DAY_COUNT))
                data[idx][1].append(sum(f > 250 for f in nums) / float(WHOLE_DAY_COUNT))
                data[idx][2].append(sum(70 <= f <= 180 for f in nums) / float(WHOLE_DAY_COUNT))
                data[idx][3].append(sum(70 <= f <= 150 for f in nums) / float(WHOLE_DAY_COUNT))
                data[idx][4].append(sum(f < 70 for f in nums) / float(WHOLE_DAY_COUNT))
                data[idx][5].append(sum(f < 54 for f in nums) / float(WHOLE_DAY_COUNT))


def process_data():
    __process_data(True)
    __process_data(False)


def write_result():
    with open(RESULT_FILENAME, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for data in [AUTO_DATA, MANUAL_DATA]:
            row = []
            for time_data in data:
                for percentage_list_data in time_data:
                    sum_val = sum(percentage_list_data)
                    if sum_val == 0:
                        row.append(0)
                    else:
                        row.append(sum_val / len(percentage_list_data))

            print('writing row: %s' % row)
            writer.writerow(row)


def __debug_print_maps():
    print('auto_daytime_map: %s' % AUTO_DAYTIME_MAP)
    print('auto_overnight_map: %s' % AUTO_OVERNIGHT_MAP)
    print('auto_wholeday_map: %s' % AUTO_WHOLEDAY_MAP)

    print('manual_daytime_map: %s' % MANUAL_DAYTIME_MAP)
    print('manual_overnight_map: %s' % MANUAL_OVERNIGHT_MAP)
    print('manual_wholeday_map: %s' % MANUAL_WHOLEDAY_MAP)


def __debug_print_result_data():
    print('auto_data: %s' % AUTO_DATA)
    print('manual_data: %s' % MANUAL_DATA)


if __name__ == '__main__':
    cgm_raw_data = pandas.read_csv(CGM_FILENAME, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
    cgm_arr = np.array(cgm_raw_data)

    last_auto_datetime = get_last_auto_mode_date_time()

    collect_data()
    # __debug_print_maps()

    process_data()
    # __debug_print_result_data()

    write_result()

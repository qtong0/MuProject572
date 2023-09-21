import csv


def write_result():
    with open('Result.csv', 'w') as result_file:
        writer = csv.writer(result_file)
        writer.writerow([0, 0, 0, 0, 0, 0])


if __name__ == '__main__':
    write_result()

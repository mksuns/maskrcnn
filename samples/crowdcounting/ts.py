# -*- coding: utf-8 -*-

import os
import time


def print_ts(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), message)


def run(interval, command):
    print_ts("-" * 100)
    print_ts("Command %s" % command)
    print_ts("Starting every %s seconds." % interval)
    print_ts("-" * 100)
    while True:
        try:
            # sleep for the remaining seconds of interval
            time_remaining = interval - time.time() % interval
            print_ts("Sleeping until %s (%s seconds)..." % ((time.ctime(time.time() + time_remaining)), time_remaining))
            time.sleep(time_remaining)
            print_ts("Starting command.")
            # execute the command
            status = os.system(command)
            print_ts("-" * 100)
            print_ts("Command status = %s." % status)
        except Exception as e:
            print(e)


# if __name__ == "__main__":
#     intervali = 5
#     commandi = r"ls"
#     run(intervali, commandi)

# print(time.strftime('%Y%m%d%H%M%S', time.localtime()))
ROOT_DIR = os.path.abspath("../../")
TEST_INPUT_DIR = os.path.join(ROOT_DIR, 'detect_client/public/test_inputs')
def delete_file(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


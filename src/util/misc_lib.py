import os
from time import gmtime, strftime
from typing import Iterable, TypeVar, Callable, Dict, List, Any, Tuple

A = TypeVar('A')
B = TypeVar('B')


def average(l):
    if len(l) == 0:
        return 0
    return sum(l) / len(l)


def threshold05(s) -> int:
    return 1 if s >= 0.5 else 0


def tprint(*arg):
    tim_str = strftime("%H:%M:%S", gmtime())
    all_text = " ".join(str(t) for t in arg)
    print("{} : {}".format(tim_str, all_text))


def exist_or_mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def validate_equal(v1, v2):
    if v1 != v2:
        print("Warning {} != {}".format(v1, v2))


def join_with_tab(l):
    return "\t".join([str(t) for t in l])


def print_table(rows):
    for row in rows:
        print(join_with_tab(row))

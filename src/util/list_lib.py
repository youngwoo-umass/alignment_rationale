from functools import reduce
from typing import Callable, TypeVar, Iterable, List, Dict, Tuple

A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


def lmap(func: Callable[[A], B],
         iterable_something: Iterable[A]) -> List[B]:
    return list([func(e) for e in iterable_something])


def lmap_pairing(func: Callable[[A], B],
         iterable_something: Iterable[A]) -> List[Tuple[A, B]]:
    return list([(e, func(e)) for e in iterable_something])


def lmap_w_exception(func: Callable[[A], B],
                     iterable_something: Iterable[A],
                     exception) -> List[B]:
    class Fail:
        pass

    def func_warp(e):
        try:
            return func(e)
        except exception:
            return Fail()
    r1 = list([func_warp(e) for e in iterable_something])
    return list([e for e in r1 if type(e) != Fail])


def l_to_map(func: Callable[[A], B],
         iterable_something: Iterable[A]) -> Dict[A, B]:
    return {k: func(k) for k in iterable_something}


def idx_where(func: Callable[[A], B],
         iterable_something: Iterable[A]) -> List[int]:
    return [idx for idx, item in enumerate(iterable_something) if func(item)]


def dict_value_map(func: Callable[[A], B], dict_like: Dict[C, A]) -> Dict[C, B]:
    return {k: func(v) for k, v in dict_like.items()}


def dict_key_map(func: Callable[[A], B], dict_like: Dict[A, C]) -> Dict[B, C]:
    return {func(k): v for k, v in dict_like.items()}


def lfilter(func: Callable[[A], B], iterable_something: Iterable[A]) -> List[A]:
    return list(filter(func, iterable_something))


def drop_none(iterable_something: Iterable[A]) -> List[A]:
    return list(filter(lambda x: x is not None, iterable_something))


def drop_empty_elem(iterable_something: Iterable[A]) -> List[A]:
    return list(filter(lambda x: x, iterable_something))


def lfilter_not(func: Callable[[A], B], iterable_something: Iterable[A]) -> List[A]:
    return list(filter(lambda x: not func(x), iterable_something))


def lreduce(initial_val: B, func: Callable[[A, B], B], iterable_something: Iterable[A]) -> List[B]:
    return list(reduce(func, iterable_something, initial_val))


def unique_from_sorted(l: Iterable[A]) -> List[A]:
    def combine(prev_list, new_elem):
        if not prev_list or prev_list[-1] != new_elem:
            return prev_list + [new_elem]
        else:
            return prev_list

    return lreduce([], combine, l)


def foreach(func, iterable_something):
    for e in iterable_something:
        func(e)


def reverse(l: Iterable[A]) -> List[A]:
    return list(reversed(l))


def flatten(z: Iterable[Iterable[A]]) -> Iterable[A]:
    return [y for x in z for y in x]


def lflatten(z: Iterable[Iterable[A]]) -> List[A]:
    return list([y for x in z for y in x])


def flatten_map(func: Callable[[A], B], z: Iterable[Iterable[A]]) -> List[B]:
    return list([func(y) for x in z for y in x])


def left(pairs: Iterable[Tuple[A, B]]) -> List[A]:
    return list([a for a, b in pairs])


def right(pairs: Iterable[Tuple[A, B]]) -> List[B]:
    return list([b for a, b in pairs])


def get_max_idx(l) -> int:
    cur_max = l[0]
    cur_max_idx = 0
    for idx, item in enumerate(l):
        if item > cur_max:
            cur_max = item
            cur_max_idx = idx

    return cur_max_idx


def find_where(func: Callable[[A], bool], z: Iterable[A]) -> A:
    t = (x for x in z if func(x))
    return next(t)


def list_join(l_list: Iterable[Iterable], sep: List) -> List:
    output = []
    is_first = True
    for idx, l in enumerate(l_list):
        if not is_first:
            output.extend(sep)
        output.extend(l)
        is_first = False

    return output


def index_by_fn(func: Callable[[A], B], z: Iterable[A]) -> Dict[B, A]:
    d_out: Dict[B, A] = {}
    for e in z:
        d_out[func(e)] = e
    return d_out


def list_equal(a: List, b: List):
    if len(a) != len(b):
        return False

    for a_e, b_e in zip(a, b):
        if a_e != b_e:
            return False
    return True


def transpose(m):
    return [[row[i] for row in m] for i in range(len(m[0]))]


class MaxKeyValue:
    def __init__(self):
        self.max_key = None
        self.max_value = None

    def update(self, k, v):
        if self.max_value is None:
            self.max_key = k
            self.max_value = v
        elif self.max_value < v:
            self.max_key = k
            self.max_value = v


def all_equal(a: List):
    v = None
    for item in a:
        if v is None:
            v = item
        else:
            if v != item:
                return False
    return True


def pairzip(l1: Iterable[A], l2: Iterable[B]) -> List[Tuple[A, B]]:
    output: List[Tuple[A, B]] = []
    for a, b in zip(l1, l2):
        output.append((a, b))
    return output


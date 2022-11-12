import time
from typing import List, Callable, Generic, Iterable, Tuple
from typing import TypeVar

A = TypeVar('A')
B = TypeVar('B')


class PromiseKeeper:
    def __init__(self,
                 list_fn: Callable[[A], B],
                 time_estimate=None
                 ):
        self.X_list = []
        self.list_fn = list_fn
        self.time_estimate = time_estimate

    def do_duty(self, log_size=False, reset=True):
        x_list = list([X.X for X in self.X_list])
        if self.time_estimate is not None:
            estimate = self.time_estimate * len(x_list)
            if estimate > 10:
                print("PromiseKeeper - Expected time: {0:.0f}sec".format(estimate))

        if log_size:
            print("PromiseKeeper - {} items".format(len(x_list)))

        st = time.time()
        y_list = self.list_fn(x_list)
        for X, y in zip(self.X_list, y_list):
            X.future().set_value(y)

        ed = time.time()
        if log_size:
            elapsed = ed - st
            s_per_inst = elapsed / len(x_list) if len(x_list) else 0
            print(f"finished in {elapsed}s. {s_per_inst} per item")

        if reset:
            self.reset()

    def get_future(self, x):
        return MyPromise(x, self).future()

    def reset(self):
        self.X_list = []

T = TypeVar('T')

class MyFuture(Generic[T]):
    def __init__(self):
        self._Y = None
        self.f_ready = False

    def get(self):
        if not self.f_ready:
            raise Exception("Future is not ready.")
        return self._Y

    def set_value(self, v):
        self._Y = v
        self.f_ready = True


class MyPromise:
    def __init__(self, X, promise_keeper: PromiseKeeper):
        self.X = X
        self.Y = MyFuture()
        promise_keeper.X_list.append(self)

    def future(self):
        return self.Y



def sum_future(futures):
    return sum([f.get() for f in futures])


def max_future(futures):
    return max([f.get() for f in futures])


# get from all element futures in the list
def list_future(futures: Iterable[MyFuture[T]]) -> List[T]:
    return list([f.get() for f in futures])


def promise_to_items(promises: List[MyPromise]):
    return list_future([p.future() for p in promises])


Parent = TypeVar("Parent")
Child = TypeVar("Child")
MiddleOutputType = TypeVar("MiddleOutputType")
OutputType = TypeVar("OutputType")


#
def parent_child_pattern(p_list: List[Parent],
                         enum_children: Callable[[Parent], List[Child]],
                         work_for_children: Callable[[List[Child]], List[OutputType]]
                         ) -> List[Tuple[Parent, List[OutputType]]]:

    pk = PromiseKeeper(work_for_children)
    future_list_list = []
    for p in p_list:
        future_list: List[MyFuture[OutputType]] = [MyPromise(c, pk).future() for c in enum_children(p)]
        future_list_list.append(future_list)

    pk.do_duty(False, False)

    output: List[Tuple[Parent, List[OutputType]]] = []
    for p, fl in zip(p_list, future_list_list):
        output_list: List[OutputType] = list_future(fl)
        output.append((p, output_list))
    return output


def parent_child_reduce_pattern(p_list: List[Parent],
                                enum_children: Callable[[Parent], List[Child]],
                                work_for_children: Callable[[List[Child]], List[MiddleOutputType]],
                                reduce_fn: Callable[[List[MiddleOutputType]], OutputType]
                                ) -> List[Tuple[Parent, OutputType]]:

    pk = PromiseKeeper(work_for_children)
    future_list_list: List[List[MyFuture[MiddleOutputType]]] = []
    for p in p_list:
        future_list: List[MyFuture[MiddleOutputType]] = [MyPromise(c, pk).future() for c in enum_children(p)]
        future_list_list.append(future_list)

    pk.do_duty(False, False)

    output: List[Tuple[Parent, OutputType]] = []
    for p, fl in zip(p_list, future_list_list):
        middle_output_list: List[MiddleOutputType] = list_future(fl)
        output: OutputType = reduce_fn(middle_output_list)
        output.append((p, output))
    return output


def list_list_pattern(list_list: List[List[Child]],
                      work_for_children: Callable[[List[Child]], List[OutputType]]
                      ) -> List[List[OutputType]]:
    pk = PromiseKeeper(work_for_children)
    future_list_list = []
    for l in list_list:
        future_list: List[MyFuture[OutputType]] = [MyPromise(c, pk).future() for c in l]
        future_list_list.append(future_list)

    pk.do_duty(False, False)

    output: List[List[OutputType]] = []
    for fl in future_list_list:
        output_list: List[OutputType] = list_future(fl)
        output.append(output_list)
    return output


if __name__ == '__main__':
    def list_fn(l):
        r = []
        for i in l:
            r.append(i * 2)
            time.sleep(1)
        return r


    pk = PromiseKeeper(list_fn)
    X_list = list(range(10))
    y_list = []
    for x in X_list:
        y = MyPromise(x, pk).future()
        y_list.append(y)

    pk.do_duty()

    for e in y_list:
        print(e.Y)


from typing import List, Callable, Dict, TypeVar, Generic

import sqlalchemy.exc
from sqlalchemy.orm import sessionmaker

from datastore.cache_sql import get_engine_from_sqlite_path, CacheTableS
from datastore.cached_client import MemoryCachedClient
from util.list_lib import right, left
from datastore.cache_db import bulk_save, read_cache_from_sqlite, build_db, \
    read_cache_s_from_sqlite, build_db_s, bulk_save_s, read_cache_str_from_sqlite, fetch_by_key

T = TypeVar('T')
V = TypeVar('V')


class SQLBasedCacheClient(Generic[T, V]):
    def __init__(self,
                 forward_fn: Callable[[List[T]], List[V]],
                 hash_fn: Callable[[T], str],
                 overhead_per_item=None,
                 save_path=None,
                 save_interval=100):
        try:
            dictionary: Dict[str, V] = read_cache_from_sqlite(save_path)
        except sqlalchemy.exc.OperationalError:
            print("Initializing Database")
            build_db(save_path)
            dictionary: Dict[str, V] = {}

        def overhead_calc(items: List[T]) -> float:
            if overhead_per_item is None:
                return len(items) * (0.035 * 2)
            else:
                return len(items) * overhead_per_item

        self.volatile_cache_client: MemoryCachedClient = MemoryCachedClient(
            forward_fn,
            hash_fn,
            dictionary,
            overhead_calc
        )
        self.save_path = save_path
        self.save_per_prediction = save_interval

    def predict(self, segs: List[T]) -> List[V]:
        ret: List[V] = self.volatile_cache_client.predict(segs)
        n_new = len(self.volatile_cache_client.get_new_items())
        if self.save_per_prediction <= n_new:
            self.save_cache()
        return ret

    def save_cache(self):
        bulk_save(self.save_path, self.volatile_cache_client.get_new_items())
        self.volatile_cache_client.reset_new_items()

    def get_last_overhead(self):
        return self.volatile_cache_client.get_last_overhead()


class SQLBasedCacheClientS(Generic[T, V]):
    def __init__(self,
                 forward_fn: Callable[[List[T]], List[V]],
                 hash_fn: Callable[[T], str],
                 overhead_per_item=None,
                 save_path=None,
                 save_interval=100):
        try:
            dictionary: Dict[str, V] = read_cache_s_from_sqlite(save_path)
        except sqlalchemy.exc.OperationalError:
            print("Initializing Database")
            build_db_s(save_path)
            dictionary: Dict[str, V] = {}

        def overhead_calc(items: List[T]) -> float:
            if overhead_per_item is None:
                return len(items) * (0.035 * 2)
            else:
                return len(items) * overhead_per_item

        self.volatile_cache_client: MemoryCachedClient = MemoryCachedClient(
            forward_fn,
            hash_fn,
            dictionary,
            overhead_calc
        )
        self.save_path = save_path
        self.save_per_prediction = save_interval

    def predict(self, segs: List[T]) -> List[V]:
        ret: List[V] = self.volatile_cache_client.predict(segs)
        n_new = len(self.volatile_cache_client.get_new_items())
        if self.save_per_prediction <= n_new:
            self.save_cache()
        return ret

    def save_cache(self):
        bulk_save_s(self.save_path, self.volatile_cache_client.get_new_items())
        self.volatile_cache_client.reset_new_items()

    def get_last_overhead(self):
        return self.volatile_cache_client.get_last_overhead()

# Do not read DB at start up
# If queried key is not found in MemoryCachedClient, check db,
#   if not found in db, directly calculate

class SQLBasedLazyCacheClientS(Generic[T, V]):
    def __init__(self,
                 forward_fn: Callable[[List[T]], List[V]],
                 hash_fn: Callable[[T], str],
                 overhead_per_item=None,
                 save_path=None,
                 save_interval=100):
        dictionary: Dict[str, V] = {}

        def overhead_calc(items: List[T]) -> float:
            if overhead_per_item is None:
                return len(items) * (0.035 * 2)
            else:
                return len(items) * overhead_per_item

        self.volatile_cache_client: MemoryCachedClient = MemoryCachedClient(
            self._predict_w_db_check,
            hash_fn,
            dictionary,
            overhead_calc
        )
        self.save_path = save_path
        self.save_per_prediction = save_interval
        self.forward_fn = forward_fn
        self.new_item_dict: Dict[str, V] = {}
        self.hash_fn = hash_fn

    def _predict_w_db_check(self, items: List[T]) -> List[V]:
        # check db
        todo = []
        result_d: Dict[int, V] = {}
        engine = get_engine_from_sqlite_path(self.save_path)
        session_maker = sessionmaker(bind=engine)
        session = session_maker()

        for i, seg in enumerate(items):
            try:
                key = self.hash_fn(seg)
                result: V = fetch_by_key(session, CacheTableS, key)
                result_d[i] = result
            except KeyError:
                todo.append((i, seg))

        items_to_predict: List[T] = right(todo)
        results: List[V] = self._direct_predict(items_to_predict)
        for i, result in zip(left(todo), results):
            result_d[i] = result

        return [result_d[i] for i in range(len(items))]

    def _direct_predict(self, items: List[T]) -> List[V]:
        results: List[V] = self.forward_fn(items)
        return results

    def predict(self, segs: List[T]) -> List[V]:
        ret: List[V] = self.volatile_cache_client.predict(segs)
        n_new = len(self.new_item_dict)
        if self.save_per_prediction <= n_new:
            self.save_cache()
        return ret

    def save_cache(self):
        bulk_save_s(self.save_path, self.new_item_dict)
        self.new_item_dict = {}

    def get_last_overhead(self):
        return self.volatile_cache_client.get_last_overhead()


class SQLBasedCacheClientStr(Generic[T]):
    def __init__(self,
                 forward_fn: Callable[[List[T]], List[str]],
                 hash_fn: Callable[[T], str],
                 overhead_per_item=None,
                 save_path=None,
                 save_interval=100):
        try:
            dictionary: Dict[str, str] = read_cache_str_from_sqlite(save_path)
        except sqlalchemy.exc.OperationalError:
            print("Initializing Database")
            build_db_s(save_path)
            dictionary: Dict[str, str] = {}

        def overhead_calc(items: List[T]) -> float:
            if overhead_per_item is None:
                return len(items) * (0.035 * 2)
            else:
                return len(items) * overhead_per_item

        self.volatile_cache_client: MemoryCachedClient = MemoryCachedClient(
            forward_fn,
            hash_fn,
            dictionary,
            overhead_calc
        )
        self.save_path = save_path
        self.save_per_prediction = save_interval

    def predict(self, segs: List[T]) -> List[str]:
        ret: List[V] = self.volatile_cache_client.predict(segs)
        n_new = len(self.volatile_cache_client.get_new_items())
        if self.save_per_prediction <= n_new:
            self.save_cache()
        return ret

    def save_cache(self):
        bulk_save_s(self.save_path, self.volatile_cache_client.get_new_items())
        self.volatile_cache_client.reset_new_items()

    def get_last_overhead(self):
        return self.volatile_cache_client.get_last_overhead()
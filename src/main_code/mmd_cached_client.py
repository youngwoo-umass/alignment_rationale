from typing import List, Callable

from sqlalchemy.orm import sessionmaker

from cpath import at_output_dir, get_cache_sqlite_path
from datastore.cache_sql import get_engine_from_sqlite_path, CacheTableF
from datastore.sql_based_cache_client import SQLBasedCacheClient

from main_code.data_structure.segmented_instance.seg_instance import SegmentedInstance
from main_code.mmd_z_client import get_mmd_client_wrap
from msmarco_model.mmd_z_direct import get_mmd_direct_wrap

FUNC_SIG = Callable[[List[SegmentedInstance]], List[float]]


def get_mmd_client(option: str) -> Callable[[List[SegmentedInstance]], List[float]]:
    if option == "localhost":
        return get_mmd_client_wrap()
    elif option == "direct":
        print("use direct predictor")
        return get_mmd_direct_wrap()
    else:
        raise ValueError


def get_mmd_cache_client(option, hooking_fn=None) -> SQLBasedCacheClient:
    forward_fn_raw: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(option)
    if hooking_fn is not None:
        def forward_fn(items: List[SegmentedInstance]) -> List[float]:
            hooking_fn(items)
            return forward_fn_raw(items)
    else:
        forward_fn = forward_fn_raw

    cache_client = SQLBasedCacheClient(forward_fn,
                                       SegmentedInstance.str_hash,
                                       0.035,
                                       get_cache_sqlite_path())
    return cache_client


def db_test():
    sqlite_path = at_output_dir("db", "temp.db")
    # build_db(sqlite_path)
    engine = get_engine_from_sqlite_path(sqlite_path)
    key = "something"
    value = 0.42
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    e = CacheTableF(key=key, value=value)
    session.add(e)
    session.flush()
    session.commit()


def get_engine():
    engine = get_engine_from_sqlite_path(get_cache_sqlite_path())
    return engine


import ast
import time
from typing import Dict
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound

from cpath import get_cache_sqlite_path
from datastore.cache_sql import get_engine_from_sqlite_path, CacheTableF, Base, index_table, CacheTableS


def has_key(session, table_class, key):
    try:
        q_res = session.query(table_class).filter(table_class.key == key).one()
        return True
    except NoResultFound as e:
        return False


def fetch_by_key(session, table_class, key):
    try:
        q_res = session.query(table_class).filter(table_class.key == key).one()
        v = ast.literal_eval(q_res.value)
        return v
    except NoResultFound as e:
        raise KeyError


def fetch_by_key_auto(sqlite_path, key):
    engine = get_engine_from_sqlite_path(sqlite_path)
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    return fetch_by_key(session, CacheTableS, key)


def bulk_save(sqlite_path, key_and_value_list):
    # tprint("bulk_save ENTRY")
    engine = get_engine_from_sqlite_path(sqlite_path)
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    for key, value in key_and_value_list.items():
        if not has_key(session, CacheTableF, key):
            e = CacheTableF(key=key, value=value)
            session.add(e)
            session.flush()
    session.commit()
    # tprint("bulk_save EXIT")


def bulk_save_s(sqlite_path, key_and_value_dict):
    # tprint("bulk_save ENTRY")
    engine = get_engine_from_sqlite_path(sqlite_path)
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    for key, value in key_and_value_dict.items():
        if not has_key(session, CacheTableS, key):
            value_s = str(value)
            e = CacheTableS(key=key, value=value_s)
            session.add(e)
            session.flush()
    session.commit()


def read_cache_from_sqlite(sqlite_path) -> Dict:
    st = time.time()
    engine = get_engine_from_sqlite_path(sqlite_path)
    session_maker = sessionmaker(bind=engine)
    with session_maker() as session:
        q_res_itr = session.query(CacheTableF).all()
        d = {}
        for row in q_res_itr:
            d[row.key] = row.value
        ed = time.time()
        print("read_cache_from_sqlite() - {0} items read at {1:.2f}sec".format(len(d), ed - st))
    return d


def read_cache_s_from_sqlite(sqlite_path) -> Dict:
    st = time.time()
    engine = get_engine_from_sqlite_path(sqlite_path)
    session_maker = sessionmaker(bind=engine)
    with session_maker() as session:
        q_res_itr = session.query(CacheTableS).all()
        d = {}
        for row in q_res_itr:
            s = row.value
            d[row.key] = ast.literal_eval(s)
        ed = time.time()
        print("read_cache_s_from_sqlite() - {0} items read at {1:.2f}sec".format(len(d), ed - st))
    return d


def read_cache_str_from_sqlite(sqlite_path) -> Dict:
    st = time.time()
    engine = get_engine_from_sqlite_path(sqlite_path)
    session_maker = sessionmaker(bind=engine)
    with session_maker() as session:
        q_res_itr = session.query(CacheTableS).all()
        d = {}
        for row in q_res_itr:
            s = row.value
            d[row.key] = s
        ed = time.time()
        print("read_cache_str_from_sqlite() - {0} items read at {1:.2f}sec".format(len(d), ed - st))
    return d


def build_db(sqlite_path):
    engine = get_engine_from_sqlite_path(sqlite_path)
    Base.metadata.create_all(engine)
    index_table(CacheTableF, engine)


def build_db_s(sqlite_path):
    engine = get_engine_from_sqlite_path(sqlite_path)
    Base.metadata.create_all(engine)
    index_table(CacheTableS, engine)



def main():
    build_db(get_cache_sqlite_path())


if __name__ == "__main__":
    main()

from sqlalchemy import Column, String, Index, REAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

Base = declarative_base()


class KeyValueBase(object):
    key = Column('key', String(), primary_key=True, index=True, sqlite_on_conflict_not_null='IGNORE')
    value = Column('value', REAL)


class CacheTableF(Base, KeyValueBase):
    __tablename__ = "cache"


class KeyValueSBase(object):
    key = Column('key', String(), primary_key=True, index=True, sqlite_on_conflict_not_null='IGNORE')
    value = Column('value', String())


class CacheTableS(Base, KeyValueSBase):
    __tablename__ = "cacheS"


def get_engine_from_sqlite_path(sqlite_path):
    sqlite_path = 'sqlite:///' + sqlite_path
    engine = create_engine(sqlite_path)
    return engine


def index_table(table, engine):
    index_ = Index(table.__tablename__ + '__index', table.key)
    index_.create(bind=engine)

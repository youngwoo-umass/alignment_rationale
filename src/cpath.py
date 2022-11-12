import os
from os.path import dirname

from util.misc_lib import exist_or_mkdir

project_root = os.path.abspath(dirname(dirname((os.path.abspath(__file__)))))
data_path = os.path.join(project_root, 'data')
src_path = os.path.join(project_root, 'src')
exist_or_mkdir(data_path)
cache_path = os.path.join(data_path, 'cache')
exist_or_mkdir(cache_path)
json_cache_path = os.path.join(data_path, 'json')
exist_or_mkdir(json_cache_path)
output_path = os.path.join(project_root, 'output')
model_path = os.path.join(project_root, 'model')


def at_output_dir(folder_name, file_name):
    return os.path.join(output_path, folder_name, file_name)


def at_data_dir(folder_name, file_name):
    return os.path.join(data_path, folder_name, file_name)


def get_cache_sqlite_path():
    return at_output_dir("db", "mmd_cache.sqlite")


def get_mmd_model_path():
    return model_path


MMD_PORT = 8129

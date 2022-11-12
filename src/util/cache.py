import json
from typing import TypeVar


def save_list_to_jsonl(item_list, save_path):
    f = open(save_path, "w")
    for item in item_list:
        s = json.dumps(item)
        f.write(s + "\n")
    f.close()


def save_list_to_jsonl_w_fn(item_list, save_path, to_json):
    f = open(save_path, "w")
    for item in item_list:
        s = json.dumps(to_json(item))
        f.write(s + "\n")
    f.close()


def load_list_from_jsonl(save_path, from_json):
    f = open(save_path, "r")
    return [from_json(json.loads(line)) for line in f]


def named_tuple_to_json(obj):
    _json = {}
    if isinstance(obj, tuple):
        datas = obj._asdict()
        for data in datas:
            _json[data] = (datas[data])
    return _json


T = TypeVar('T')

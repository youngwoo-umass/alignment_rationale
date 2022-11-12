import json
import os
from typing import List

from main_code.data_structure.ds_helper import parse_related_eval_answer_from_json, \
    parse_related_binary_answer_from_json
from main_code.data_structure.eval_data_structure import Alignment2D, RelatedBinaryAnswer
from cpath import output_path
from util.misc_lib import exist_or_mkdir


def get_related_save_path(method):
    dir_path = os.path.join(output_path, "related_scores")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, "{}.score".format(method))
    return save_path


def get_related_binary_save_path(method):
    dir_path = os.path.join(output_path, "binary_related_scores")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, "{}.score".format(method))
    return save_path


def save_related_eval_answer(answers: List[Alignment2D], method):
    save_path = get_related_save_path(method)
    json.dump(answers, open(save_path, "w"), indent=True)


def load_related_eval_answer(method) -> List[Alignment2D]:
    score_path = get_related_save_path(method)
    raw_json = json.load(open(score_path, "r"))
    return parse_related_eval_answer_from_json(raw_json)


def save_json_at(data, save_path):
    json.dump(data, open(save_path, "w"), indent=True)


def load_binary_related_eval_answer(method) -> List[RelatedBinaryAnswer]:
    score_path = get_related_binary_save_path(method)
    raw_json = json.load(open(score_path, "r"))
    return parse_related_binary_answer_from_json(raw_json)


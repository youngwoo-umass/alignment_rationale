import functools
import sys
from typing import List

from main_code.data_structure.eval_data_structure import Alignment2D, RelatedBinaryAnswer, \
    convert_answer
from main_code.related_answer_data_path_helper import load_related_eval_answer, \
    get_related_binary_save_path, save_json_at


def discretize_and_save(method):
    answers: List[Alignment2D] = load_related_eval_answer(method)
    cutoff = 0.5

    def convert(score):
        if score >= cutoff:
            return 1
        else:
            return 0

    convert_answer_fn = functools.partial(convert_answer, convert)
    new_answers: List[RelatedBinaryAnswer] = list(map(convert_answer_fn, answers))
    save_path = get_related_binary_save_path(method)
    save_json_at(new_answers, save_path)


def main():
    method = sys.argv[2]
    discretize_and_save(method)


if __name__ == "__main__":
    main()

from typing import List, Tuple, Optional

from util.list_lib import flatten, right
from main_code.data_structure.eval_data_structure import RelatedBinaryAnswer
from main_code.eval_score_dp_helper import load_eval_result_b_all
from main_code.related_answer_data_path_helper import load_binary_related_eval_answer
from main_code.method_preference_count import count_paired_comparison


def get_score_for_method(method, metric) -> List[float]:
    run_name = "{}_{}".format(method, metric)
    eval_res: List[Tuple[str, List[Optional[float]]]] = load_eval_result_b_all(run_name)
    return list(flatten(right(eval_res)))


def get_prediction_for_method(method) -> List[List[int]]:
    answers: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(method)
    return list(flatten(right(answers)))


def main():
    method_list = ["exact_match", "random"]
    metric_list = ["deletion_ness_binary", "substitution_ness_binary",
                   "deletion_suff_binary", "substitution_suff_binary",
                   "deletion_ness_soft", "substitution_ness_soft",
                   "deletion_suff_soft", "substitution_suff_soft",
                   ]

    count_paired_comparison(method_list, metric_list,
                            get_score_for_method,
                            get_prediction_for_method)


if __name__ == "__main__":
    main()

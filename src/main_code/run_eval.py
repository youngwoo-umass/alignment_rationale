import random
import sys
from typing import List

from cpath import at_data_dir
from main_code.data_structure.eval_data_structure import RelatedBinaryAnswer, \
    PerProblemEvalResult, UnexpectedPolicyException
from main_code.data_structure.related_eval_instance import RelatedEvalInstance
from main_code.metric_common.segment_modify_fn import TenStepRandomSubstitutionPolicy, \
    TenStepRandomDropPolicy
from main_code.eval_metrics.deletion_metrics import DeletionNess, DeletionSuff
from main_code.eval_metrics.related_eval import run_eval
from main_code.eval_metrics.eval_metric_common import RelatedEvalMetricIF, RelatedMetric
from main_code.eval_metrics.substitution_metrics import SubstitutionNess, SubstitutionSuff
from main_code.eval_score_dp_helper import save_eval_result_b, get_run_name
from main_code.loader import load_mmde_problem
from main_code.mmd_cached_client import get_mmd_cache_client
from main_code.related_answer_data_path_helper import load_binary_related_eval_answer
from main_code.tokenizer_wo_tf import get_tokenizer


def get_100_random_spans() -> List[List[int]]:
    save_path = at_data_dir("align", "msmarco_random_spans.txt")
    spans = []
    for line in open(save_path, "r"):
        spans.append(line.strip())

    tokenizer = get_tokenizer()

    spans_as_ids = []
    for span in spans:
        ngram_ids: List[int] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(span))
        spans_as_ids.append(ngram_ids)
    return spans_as_ids


def hooking_log(items):
    n_item = len(items)
    time_estimate = 0.035 * n_item
    if time_estimate > 100:
        n_min = int(time_estimate / 60)
        print("hooking_log: requesting {} items {} min expected".format(n_item, n_min))


def get_substitution(model_interface, policy_name) -> RelatedMetric:
    client = get_mmd_cache_client(model_interface, hooking_log)
    forward_fn = client.predict
    replacee_span_list: List[List[int]] = get_100_random_spans()

    def get_word_pool(_) -> List[List[int]]:
        return replacee_span_list

    if policy_name == "substitution_ness_soft":
        return SubstitutionNess(forward_fn, TenStepRandomSubstitutionPolicy(), get_word_pool)
    elif policy_name == "substitution_ness_binary":
        return SubstitutionNess(forward_fn, TenStepRandomSubstitutionPolicy(True), get_word_pool)
    if policy_name == "substitution_suff_soft":
        return SubstitutionSuff(forward_fn, False, get_word_pool)
    elif policy_name == "substitution_suff_binary":
        return SubstitutionSuff(forward_fn, True, get_word_pool)
    else:
        raise ValueError


def get_deletion_ness(model_interface, discretize) -> DeletionNess:
    client = get_mmd_cache_client(model_interface, hooking_log)
    forward_fn = client.predict
    return DeletionNess(forward_fn, TenStepRandomDropPolicy(discretize))


def get_deletion_sufficiency(model_interface, discretize) -> DeletionSuff:
    client = get_mmd_cache_client(model_interface, hooking_log)
    forward_fn = client.predict
    return DeletionSuff(forward_fn, discretize)


def run_eval_for_method_policy(method, policy_name, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mmde_problem()
    answers: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(method)
    assert len(problems) == len(answers)
    eval_policy = get_policy(model_interface, policy_name)

    scores: List[PerProblemEvalResult] = run_eval(answers, problems, eval_policy)
    run_name = get_run_name(method, policy_name)
    save_eval_result_b(scores, run_name)


def get_policy(model_interface, policy_name) -> RelatedEvalMetricIF:
    random.seed(0)
    if policy_name.startswith("substitution_"):
        eval_policy: RelatedEvalMetricIF = get_substitution(model_interface, policy_name)
    elif policy_name == "deletion_ness_soft":
        eval_policy: RelatedEvalMetricIF = get_deletion_ness(model_interface, False)
    elif policy_name == "deletion_ness_binary":
        eval_policy: RelatedEvalMetricIF = get_deletion_ness(model_interface, True)
    elif policy_name == "deletion_suff_soft":
        eval_policy: RelatedEvalMetricIF = get_deletion_sufficiency(model_interface, False)
    elif policy_name == "deletion_suff_binary":
        eval_policy: RelatedEvalMetricIF = get_deletion_sufficiency(model_interface, True)
    else:
        raise UnexpectedPolicyException(policy_name)
    return eval_policy


def main():
    method = sys.argv[1]
    policy_name = sys.argv[2]
    if len(sys.argv) > 3:
        model_interface = sys.argv[3]
    else:
        model_interface = "localhost"

    run_eval_for_method_policy(method, policy_name, model_interface)


if __name__ == "__main__":
    main()


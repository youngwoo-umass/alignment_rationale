import random

from util.misc_lib import average
from typing import Callable, List

from main_code.data_structure.segmented_instance.segmented_text import SegmentedText, get_replaced_segment
from main_code.ep_common import DropSamplePolicyIF, ReplaceSamplePolicyIF

DocModFunc = Callable[[SegmentedText, List[int]], SegmentedText]
DocReplaceFunc = Callable[[SegmentedText, List[int], List[int]], SegmentedText]


def get_drop_zero() -> DocModFunc:
    def drop_zero(text: SegmentedText, scores: List[int]) -> SegmentedText:
        seg_len = text.get_seg_len()
        if len(scores) != seg_len:
            print("Score has {} items while text has {} segments".format(len(scores), seg_len))

        drop_indices = [idx for idx, s in enumerate(scores) if s == 0]
        new_text = text.get_dropped_text(drop_indices)
        return new_text
    return drop_zero


def assert_float_or_int(v):
    assert type(v) == float or type(v) == int


def get_replace_non_zero() -> DocReplaceFunc:
    def replace_non_zero(text: SegmentedText, scores: List[int], word: List[int]) -> SegmentedText:
        assert_float_or_int(scores[0])
        drop_indices = [idx for idx, s in enumerate(scores) if s == 1]
        return get_replaced_segment(text, drop_indices, word)
    return replace_non_zero


def get_replace_zero() -> DocReplaceFunc:
    def replace_zero(text: SegmentedText, scores: List[int], word: List[int]) -> SegmentedText:
        assert_float_or_int(scores[0])
        drop_indices = [idx for idx, s in enumerate(scores) if s == 0]
        return get_replaced_segment(text, drop_indices, word)
    return replace_zero


def get_partial_text_as_segment(text1: SegmentedText, target_seg_idx: int) -> SegmentedText:
    tokens = text1.get_tokens_for_seg(target_seg_idx)
    return SegmentedText.from_tokens_ids(tokens)


def ceil(f):
    eps = 1e-8
    return int(f - eps) + 1


class TenStepRandomDropPolicy(DropSamplePolicyIF):
    def __init__(self, discretize=False):
        self.discretize = discretize

    def get_drop_docs(self, text: SegmentedText, score_list: List[int]) -> List[SegmentedText]:
        n_pos = sum(score_list)
        segment_list = []
        for i in range(10):
            drop_rate = 1 - i / 10
            n_drop = ceil(n_pos * drop_rate)
            true_indices = [i for i in range(len(score_list)) if score_list[i]]
            n_drop = min(len(true_indices), n_drop)
            drop_indices = random.sample(true_indices, n_drop)
            if drop_indices:
                segment = text.get_dropped_text(drop_indices)
                segment_list.append(segment)
        return segment_list

    def combine_results(self, outcome_list: List[float]):
        if self.discretize:
            outcome_b = [1 if s >= 0.5 else 0 for s in outcome_list]
            return average(outcome_b)
        else:
            return average(outcome_list)


class TenStepRandomSubstitutionPolicy(ReplaceSamplePolicyIF):
    def __init__(self, discretize=False):
        self.discretize = discretize

    def get_replaced_docs(self, text: SegmentedText, score_list: List[int], word: List[int]) -> List[SegmentedText]:
        n_pos = sum(score_list)
        segment_list = []
        for i in range(10):
            drop_rate = 1 - i / 10
            n_drop = int(n_pos * drop_rate)
            true_indices = [i for i in range(len(score_list)) if score_list[i]]
            n_drop = min(len(true_indices), n_drop)
            drop_indices = random.sample(true_indices, n_drop)
            if drop_indices:
                segment = get_replaced_segment(text, drop_indices, word)
                segment_list.append(segment)
        return segment_list

    def combine_results(self, outcome_list: List[float]):
        if self.discretize:
            outcome_b = [1 if s >= 0.5 else 0 for s in outcome_list]
            return average(outcome_b)
        else:
            return average(outcome_list)

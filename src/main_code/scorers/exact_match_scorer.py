from typing import List

from main_code.data_structure.matrix_scorer_if import MatrixScorerIF, ContributionSummary
from main_code.data_structure.segmented_instance.seg_instance import SegmentedInstance
from util.list_lib import list_equal


class TokenExactMatchScorer(MatrixScorerIF):
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        l1 = inst.text1.get_seg_len()
        l2 = inst.text2.get_seg_len()

        output_table = []
        for i1 in range(l1):
            scores_per_seg = []
            tokens = inst.text1.get_tokens_for_seg(i1)
            for i2 in range(l2):
                score = 0
                for i2_i in inst.text2.get_tokens_for_seg(i2):
                    if i2_i in tokens:
                        score += 1
                scores_per_seg.append(score)
            output_table.append(scores_per_seg)

        return ContributionSummary(output_table)


class SegmentExactMatchScorer(MatrixScorerIF):
    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        l1 = inst.text1.get_seg_len()
        l2 = inst.text2.get_seg_len()

        output_table = []
        for i1 in range(l1):
            scores_per_seg: List[float] = []
            tokens1: List[int] = inst.text1.get_tokens_for_seg(i1)
            for i2 in range(l2):
                tokens2: List[int] = inst.text2.get_tokens_for_seg(i2)
                score = 1 if list_equal(tokens1, tokens2) else 0
                scores_per_seg.append(score)
            output_table.append(scores_per_seg)

        return ContributionSummary(output_table)



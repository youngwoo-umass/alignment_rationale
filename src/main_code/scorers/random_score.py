import numpy as np

from main_code.data_structure.matrix_scorer_if import ContributionSummary, MatrixScorerIF
from main_code.data_structure.segmented_instance.seg_instance import SegmentedInstance


class RandomScorer(MatrixScorerIF):
    def __init__(self):
        np.random.seed(0)

    def eval_contribution(self, inst: SegmentedInstance) -> ContributionSummary:
        l1 = inst.text1.get_seg_len()
        l2 = inst.text2.get_seg_len()
        scores = np.random.random([l1, l2])
        table = inst.score_np_table_to_table(scores)
        return ContributionSummary(table)


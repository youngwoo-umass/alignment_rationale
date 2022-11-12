import abc
from typing import List
from main_code.data_structure.segmented_instance.segmented_text import SegmentedText


class DropSamplePolicyIF(abc.ABC):
    @abc.abstractmethod
    def get_drop_docs(self,
                          text: SegmentedText,
                          score_list: List[int],
                          ) -> List[SegmentedText]:
        pass

    @abc.abstractmethod
    def combine_results(self, outcome_list: List[float]):
        pass


class ReplaceSamplePolicyIF(abc.ABC):
    @abc.abstractmethod
    def get_replaced_docs(self,
                          text: SegmentedText,
                          score_list: List[int],
                          word: List[int]) -> List[SegmentedText]:
        pass

    @abc.abstractmethod
    def combine_results(self, outcome_list: List[float]):
        pass


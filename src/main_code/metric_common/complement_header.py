import abc
from typing import List, Tuple, NamedTuple

from main_code.data_structure.segmented_instance.seg_instance import SegmentedInstance
from main_code.data_structure.segmented_instance.segmented_text import SegmentedText
from main_code.tokenizer_wo_tf import ids_to_text


class PartialSegment:
    def __init__(self, data, n_seg):
        self.data = data
        self.n_seg = n_seg

    @classmethod
    def init_one_piece(cls, tokens: List[int]):
        return PartialSegment(tokens, 1)

    @classmethod
    def init_two_piece(cls, tokens: Tuple[List[int], List[int]]):
        return PartialSegment(tokens, 2)

    def to_text(self, tokenizer) -> str:
        if self.n_seg == 1:
            return ids_to_text(tokenizer, self.data)
        elif self.n_seg == 2:
            head, tail = self.data
            return ids_to_text(tokenizer, head) + " [MASK] " + ids_to_text(tokenizer, tail)
        else:
            raise Exception("n_seg > 2 is not expected")

    def to_json(self) -> Tuple[List[int], List[int]]:
        if self.n_seg == 1:
            return self.data, []
        elif self.n_seg == 2:
            return self.data

    @classmethod
    def from_json(cls, j):
        return PartialSegment(j, 2)


class SegJoinPolicyIF(abc.ABC):
    @abc.abstractmethod
    def join_tokens(self, si: SegmentedText, new_tokens: PartialSegment, preserve_seg_idx):
        pass


class ComplementSearchOutput(NamedTuple):
    problem_id: str
    target_seg_idx: int
    complement_list: List[PartialSegment]

    def to_json(self):
        complement_list_j = [c.to_json() for c in self.complement_list]
        return {
            'problem_id': self.problem_id,
            'target_seg_idx': self.target_seg_idx,
            'complement_list': complement_list_j
        }

    @classmethod
    def from_json(cls, j):
        return ComplementSearchOutput(j['problem_id'],
                             j['target_seg_idx'],
                             list(map(PartialSegment.from_json, j['complement_list']))
                             )

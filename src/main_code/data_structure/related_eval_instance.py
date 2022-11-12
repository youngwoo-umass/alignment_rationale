from typing import NamedTuple

from main_code.data_structure.segmented_instance.seg_instance import SegmentedInstance
from main_code.data_structure.segmented_instance.segmented_text import text_to_word_level_segmented_text


class RelatedEvalInstance(NamedTuple):
    problem_id: str
    seg_instance: SegmentedInstance
    score: float

    @classmethod
    def from_json(cls, j):
        return RelatedEvalInstance(j['problem_id'],
                                   SegmentedInstance.from_json(j['seg_instance']),
                                   j['score']
                                   )

    def to_json(self):
        return {
            'problem_id': self.problem_id,
            'seg_instance': self.seg_instance.to_json(),
            'score': self.score
        }


class TextPair(NamedTuple):
    text_pair_id: str
    query_like: str
    doc_like: str

    def get_query_like(self):
        return self.query_like

    def get_doc_like(self):
        return self.doc_like


def get_word_level_rei(tokenizer, e: TextPair) -> RelatedEvalInstance:
    problem_id = e.text_pair_id
    q_seg = text_to_word_level_segmented_text(e.get_query_like(), tokenizer)
    d_seg = text_to_word_level_segmented_text(e.get_doc_like(), tokenizer)
    si = SegmentedInstance(q_seg, d_seg)
    rei = RelatedEvalInstance(problem_id, si, score=0)
    return rei
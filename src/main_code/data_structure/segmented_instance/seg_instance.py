from collections import defaultdict
import numpy as np
from typing import NamedTuple, Iterator, Dict, Tuple, Callable, List, Any


from main_code.data_structure.segmented_instance.segmented_text import SegmentedText


class SegmentedInstance(NamedTuple):
    text1: SegmentedText
    text2: SegmentedText

    def enum_seg_indice_pairs(self):
        for seg1_idx in range(self.text1.get_seg_len()):
            for seg2_idx in range(self.text2.get_seg_len()):
                yield seg1_idx, seg2_idx

    def get_drop_mask(self, seg1_idx, seg2_idx) -> np.array:
        drop_mask_per_q_seg = self.text2.get_empty_seg_mask()
        drop_mask_per_q_seg[seg2_idx] = 1
        drop_mask = [self.text2.get_empty_seg_mask() for _ in range(self.text1.get_seg_len())]
        drop_mask[seg1_idx] = drop_mask_per_q_seg
        drop_mask = np.stack(drop_mask)
        return drop_mask

    def get_empty_mask(self) -> np.ndarray:
        return np.stack([self.text2.get_empty_seg_mask() for _ in range(self.text1.get_seg_len())])

    def enum_token_idx_from_seg1_idx(self, seg_idx) -> Iterator[int]:
        yield from self.text1.seg_token_indices[seg_idx]

    def enum_token_idx_from_seg2_idx(self, seg_idx) -> Iterator[int]:
        yield from self.text2.seg_token_indices[seg_idx]

    def translate_mask(self, drop_mask: np.array) -> Dict[Tuple[int, int], int]:
        new_mask = {}
        for q_seg_idx in range(self.text1.get_seg_len()):
            for d_seg_idx in range(self.text2.get_seg_len()):
                v = drop_mask[q_seg_idx, d_seg_idx]
                if v:
                    for q_token_idx in self.text1.enum_token_idx_from_seg_idx(q_seg_idx):
                        for d_token_idx in self.enum_token_idx_from_seg2_idx(d_seg_idx):
                            k = q_token_idx, d_token_idx
                            new_mask[k] = int(v)
        return new_mask

    def translate_mask_d(self, drop_mask) -> Dict[Tuple[int, int], int]:
        new_mask = {}
        for q_seg_idx in range(self.text1.get_seg_len()):
            for d_seg_idx in range(self.text2.get_seg_len()):
                try:
                    v = drop_mask[q_seg_idx, d_seg_idx]
                    if v:
                        for q_token_idx in self.text1.enum_token_idx_from_seg_idx(q_seg_idx):
                            for d_token_idx in self.enum_token_idx_from_seg2_idx(d_seg_idx):
                                k = q_token_idx, d_token_idx
                                new_mask[k] = int(v)
                except KeyError:
                    pass
        return new_mask
    def accumulate_over(self, raw_scores, accumulate_method: Callable[[List[float]], float]):
        scores_d = defaultdict(list)
        for q_seg_idx in range(self.text1.get_seg_len()):
            for d_seg_idx in range(self.text2.get_seg_len()):
                key = q_seg_idx, d_seg_idx
                for q_token_idx in self.text1.enum_token_idx_from_seg_idx(q_seg_idx):
                    for d_token_idx in self.text2.enum_token_idx_from_seg_idx(d_seg_idx):
                        v = raw_scores[q_token_idx, d_token_idx]
                        scores_d[key].append(v)

        out_d = {}
        for q_seg_idx in range(self.text1.get_seg_len()):
            for d_seg_idx in range(self.text2.get_seg_len()):
                key = q_seg_idx, d_seg_idx
                scores = scores_d[key]
                out_d[key] = accumulate_method(scores)
        return out_d

    def score_d_to_table(self, contrib_score_d: Dict[Tuple[int, int], Any]):
        return self._score_d_to_table(contrib_score_d)

    def _score_d_to_table(self, contrib_score_table):
        table = []
        for q_seg_idx in range(self.text1.get_seg_len()):
            row = []
            for d_seg_idx in range(self.text2.get_seg_len()):
                key = q_seg_idx, d_seg_idx
                row.append(contrib_score_table[key])
            table.append(row)
        return table

    def score_np_table_to_table(self, contrib_score_table):
        return self._score_d_to_table(contrib_score_table)

    def get_seg2_dropped_instances(self, drop_indices):
        return SegmentedInstance(SegmentedText(self.text1.tokens_ids, self.text1.seg_token_indices),
                                 self.text2.get_dropped_text(drop_indices),
                                 )
    @classmethod
    def from_flat_args(cls,
                       text1_tokens_ids,
                       text2_tokens_ids,
                       text1_seg_indices,
                       text2_seg_indices,
                       ):
        return SegmentedInstance(SegmentedText(text1_tokens_ids, text1_seg_indices),
                                 SegmentedText(text2_tokens_ids, text2_seg_indices),
                                 )

    def to_json(self):
        return {
            'text1': self.text1.to_json(),
            'text2': self.text2.to_json(),
        }

    @classmethod
    def from_json(cls, j):
        return SegmentedInstance(SegmentedText.from_json(j['text1']),
                                 SegmentedText.from_json(j['text2']),
                                 )

    def str_hash(self) -> str:
        return str(self.to_json())
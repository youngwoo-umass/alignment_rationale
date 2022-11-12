from typing import List, Iterable, Callable, Tuple
from typing import NamedTuple, Iterator

import numpy as np

from util.cache import named_tuple_to_json
from main_code.tokenizer_wo_tf import ids_to_text, pretty_tokens, is_continuation
from util.list_lib import list_equal


class SegmentedText(NamedTuple):
    tokens_ids: List[int]
    seg_token_indices: List[List[int]]

    def get_seg_len(self):
        return len(self.seg_token_indices)

    def enum_seg_idx(self) -> Iterator[int]:
        yield from range(self.get_seg_len())

    def get_empty_seg_mask(self):
        return np.zeros([self.get_seg_len()], np.int)

    def enum_token_idx_from_seg_idx(self, seg_idx) -> Iterator[int]:
        yield from self.seg_token_indices[seg_idx]

    def get_token_idx_as_head_tail(self, seg_idx) -> Tuple[List[int], List[int]]:
        indice = self.seg_token_indices[seg_idx]
        prev_idx = None
        split_idx = len(indice)
        for j, idx in enumerate(indice):
            if prev_idx is not None:
                if prev_idx == idx-1:
                    pass
                else:
                    split_idx = j

        return indice[:split_idx], indice[split_idx:]

    def get_tokens_for_seg(self, seg_idx) -> List[int]:
        return [self.tokens_ids[i] for i in self.seg_token_indices[seg_idx]]

    def get_seg_text(self, tokenizer, seg_idx):
        return pretty_tokens(tokenizer.convert_ids_to_tokens(self.get_tokens_for_seg(seg_idx)), True)

    def get_tokens_as_text(self, tokenizer):
        return [self.get_seg_text(tokenizer, i) for i in self.enum_seg_idx()]

    def get_dropped_text(self, drop_indices: Iterable[int]):
        new_seg = []
        new_seg_indices = []
        offset = 0
        for seg_idx in range(self.get_seg_len()):
            if seg_idx in drop_indices:
                offset = offset - len(self.seg_token_indices[seg_idx])
            else:
                for i in self.enum_token_idx_from_seg_idx(seg_idx):
                    new_seg.append(self.tokens_ids[i])

                new_indices = [idx + offset for idx in self.enum_token_idx_from_seg_idx(seg_idx)]
                new_seg_indices.append(new_indices)
        return SegmentedText(new_seg, new_seg_indices)

    def get_sliced_text(self, target_indices: List[int]):
        drop_indices = self.get_remaining_indices(target_indices)
        return self.get_dropped_text(drop_indices)

    def get_remaining_indices(self, target_indices):
        iterable: Iterable[int] = self.enum_seg_idx()
        drop_indices: List[int] = [i for i in iterable if i not in target_indices]
        return drop_indices

    def to_json(self):
        return named_tuple_to_json(self)

    @classmethod
    def from_json(cls, j):
        return SegmentedText(j['tokens_ids'], j['seg_token_indices'])

    @classmethod
    def from_tokens_ids(cls, tokens_ids):
        seg_token_indices = [[i] for i in range(len(tokens_ids))]
        return SegmentedText(tokens_ids, seg_token_indices)

    def get_readable_rep(self, tokenizer):
        s_list = []
        for i in self.enum_seg_idx():
            s_out = self.get_segment_tokens_rep(tokenizer, i)
            s_list.append(s_out)
        return " ".join(s_list)

    def get_segment_tokens_rep(self, tokenizer, i):
        s = ids_to_text(tokenizer, self.get_tokens_for_seg(i))
        s_out = f"{i}) {s}"
        return s_out


def get_replaced_segment(s: SegmentedText, drop_indices: List[int], tokens_tbi: List[int]) -> SegmentedText:
    new_tokens_added = False
    new_tokens: List[int] = []
    new_indices = []

    def append_tokens(tokens):
        st = len(new_tokens)
        ed = st + len(tokens)
        new_tokens.extend(tokens)
        new_indices.append(list(range(st, ed)))

    for i_ in s.enum_seg_idx():
        assert type(i_) == int
        i: int = i_
        if i not in drop_indices:
            tokens = s.get_tokens_for_seg(i)
            append_tokens(tokens)
        else:
            if new_tokens_added:
                pass
            else:
                new_tokens_added = True
                append_tokens(tokens_tbi)
    return SegmentedText(new_tokens, new_indices)


def print_segment_text(tokenizer, text: SegmentedText):
    for i in range(text.get_seg_len()):
        tokens = text.get_tokens_for_seg(i)
        s = ids_to_text(tokenizer, tokens)
        print("{}:\t{}".format(i, s))


def seg_to_text(tokenizer, segment: SegmentedText) -> str:
    ids: List[int] = segment.tokens_ids
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return pretty_tokens(tokens, True)


def get_highlighted_text(tokenizer, drop_indices, text2: SegmentedText):
    all_words = [ids_to_text(tokenizer, text2.get_tokens_for_seg(i)) for i in range(text2.get_seg_len())]
    for i in drop_indices:
        all_words[i] = "[{}]".format(all_words[i])
    text = " ".join(all_words)
    return text


def get_word_level_location_w_ids(tokenizer, input_ids) -> List[List[int]]:
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    intervals: List[List[int]] = []
    start = 0
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if idx == 0:
            pass
        elif is_continuation(token):
            pass
        elif token == "[PAD]":
            break
        else:
            end = idx
            intervals.append(list(range(start, end)))
            start = idx
        idx += 1
    end = idx
    if end > start:
        l: List[int] = list(range(start, end))
        intervals.append(l)
    return intervals


def word_segment_w_indices(tokenizer, input_ids) -> Tuple[List[int], List[List[int]]]:
    word_location_list = get_word_level_location_w_ids(tokenizer, input_ids)
    return input_ids, word_location_list


def get_word_level_segmented_text(tokenizer, input_ids: List[int]) -> SegmentedText:
    input_ids, word_location_list = word_segment_w_indices(tokenizer, input_ids)
    return SegmentedText(input_ids, word_location_list)


def get_word_level_segmented_text_from_str(tokenizer, s: str) -> SegmentedText:
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
    input_ids, word_location_list = word_segment_w_indices(tokenizer, input_ids)
    return SegmentedText(input_ids, word_location_list)


def token_list_to_segmented_text(tokenizer, token_list: List[str]) -> SegmentedText:
    loc = 0
    all_input_ids = []
    indices_list = []
    for token in token_list:
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
        indices = [loc+idx for idx, _ in enumerate(input_ids)]
        all_input_ids.extend(input_ids)
        indices_list.append(indices)
        loc += len(input_ids)
    return SegmentedText(all_input_ids, indices_list)



def get_segment_mapping(source: SegmentedText, subsequence: SegmentedText) -> List[int]:
    # This runs greedy mapping. If this fails I need to implement nop-greedy matching
    last_match_idx = -1
    output = []
    for i in subsequence.enum_seg_idx():
        cut_tokens = subsequence.get_tokens_for_seg(i)
        match_idx = -1
        for j in range(last_match_idx+1, source.get_seg_len()):
            if list_equal(source.get_tokens_for_seg(j), cut_tokens):
                match_idx = j

        if match_idx == -1:
            raise IndexError

        output.append(match_idx)
        last_match_idx = match_idx

    return output


def merge_subtoken_level_scores(merge_subtoken_scores: Callable[[Iterable[float]], float],
                                scores: List[float],
                                t_text: SegmentedText) -> List[float]:
    output = []
    for seg_idx in t_text.enum_seg_idx():
        scores_per_seg = [scores[idx] for idx in t_text.seg_token_indices[seg_idx]]
        score_for_seg: float = merge_subtoken_scores(scores_per_seg)
        output.append(score_for_seg)
    return output


def merge_subtoken_level_scores_2d(merge_subtoken_scores: Callable[[Iterable[float]], float],
                                   scores: List[List[float]],
                                   t_text1: SegmentedText,
                                   t_text2: SegmentedText,
                                   ) -> List[List[float]]:
    output = []
    for seg_idx1 in t_text1.enum_seg_idx():
        row = []
        for seg_idx2 in t_text2.enum_seg_idx():
            scores_per_cell = []
            for idx1 in t_text1.seg_token_indices[seg_idx1]:
                for idx2 in t_text2.seg_token_indices[seg_idx2]:
                    scores_per_cell.append(scores[idx1][idx2])
            s: float = merge_subtoken_scores(scores_per_cell)
            row.append(s)
        output.append(row)
    return output



def text_to_word_level_segmented_text(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # text
    segmented_text: SegmentedText = get_word_level_segmented_text(tokenizer, input_ids)
    return segmented_text
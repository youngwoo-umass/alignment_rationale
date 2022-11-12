import json
import os

from cpath import data_path
from main_code.data_structure.segmented_instance.seg_instance import SegmentedInstance
from main_code.data_structure.segmented_instance.segmented_text import SegmentedText
from main_code.loader import load_mmde_problem
from main_code.tokenizer_wo_tf import get_tokenizer
from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple


def main():
    problems = load_mmde_problem()
    tokenizer = get_tokenizer()

    def decode_seg_text(st: SegmentedText):
        return SegmentedText(tokenizer.convert_ids_to_tokens(st.tokens_ids), st.seg_token_indices)

    j_out_list = []
    for p in problems:
        si_decoded = SegmentedInstance(decode_seg_text(p.seg_instance.text1), decode_seg_text(p.seg_instance.text2))
        j = p.to_json()
        j['seg_instance'] = si_decoded.to_json()
        j_out_list.append(j)

    save_path = os.path.join(data_path, "align", "problems_decoded.json")
    f = open(save_path, "w")
    for j in j_out_list:
        s = json.dumps(j)
        f.write(s + "\n")
    f.close()


if __name__ == "__main__":
    main()
import os
from typing import List

from util.cache import load_list_from_jsonl
from cpath import data_path
from main_code.data_structure.related_eval_instance import RelatedEvalInstance


def load_mmde_problem() -> List[RelatedEvalInstance]:
    save_path = os.path.join(data_path, "align", "problems.json")
    items: List[RelatedEvalInstance] = load_list_from_jsonl(save_path, RelatedEvalInstance.from_json)
    return items


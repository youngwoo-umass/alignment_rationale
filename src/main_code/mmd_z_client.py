from typing import List, Callable

import numpy as np
from scipy.special import softmax

from main_code.data_structure.segmented_instance.seg_instance import SegmentedInstance
from main_code.tokenizer_wo_tf import JoinEncoder
from msmarco_model.client_lib import BERTClient
from cpath import MMD_PORT


def get_mmd_client_wrap() -> Callable[[List[SegmentedInstance]], List[float]]:
    max_seq_length = 512
    client = BERTClient("http://localhost", MMD_PORT, max_seq_length)
    join_encoder = JoinEncoder(max_seq_length)

    def query_multiple(items: List[SegmentedInstance]) -> List[float]:
        max_per_run = 10 * 1000
        if len(items) == 0:
            return []

        def encode(item: SegmentedInstance):
            return join_encoder.join(item.text1.tokens_ids, item.text2.tokens_ids)

        cursor = 0
        probs_list = []
        while cursor < len(items):
            cur_items = items[cursor:cursor+max_per_run]
            payload = list(map(encode, cur_items))
            ret = client.send_payload(payload)
            ret = np.array(ret)
            probs = softmax(ret, axis=1)[:, 1]
            probs_list.append(probs)
            cursor += max_per_run
        probs_concat = np.concatenate(probs_list, axis=0)
        return probs_concat.tolist()
    return query_multiple


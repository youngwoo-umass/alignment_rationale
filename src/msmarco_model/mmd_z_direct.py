from typing import List, Callable

import numpy as np
from scipy.special import softmax

from cpath import get_mmd_model_path
from main_code.data_structure.segmented_instance.seg_instance import SegmentedInstance
from main_code.tokenizer_wo_tf import JoinEncoder
from msmarco_model.mmd_server import PredictorClsDense


def mmd_predictor():
    save_path = get_mmd_model_path()
    from tf_v2_support import disable_eager_execution
    disable_eager_execution()

    predictor = PredictorClsDense(2, 512)
    predictor.load_model(save_path)

    def predict(payload):
        sout = predictor.predict(payload)
        return sout
    return predict


def get_mmd_direct_wrap() -> Callable[[List[SegmentedInstance]], List[float]]:
    max_seq_length = 512
    join_encoder = JoinEncoder(max_seq_length)
    predict = mmd_predictor()

    def query_multiple(items: List[SegmentedInstance]) -> List[float]:
        if len(items) == 0:
            return []

        def encode(item: SegmentedInstance):
            return join_encoder.join(item.text1.tokens_ids, item.text2.tokens_ids)
        ret = predict(list(map(encode, items)))
        ret = np.array(ret)
        probs = softmax(ret, axis=1)[:, 1]
        return probs
    return query_multiple


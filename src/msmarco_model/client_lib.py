import os
import xmlrpc.client
from typing import List, Tuple

from cpath import data_path
from main_code.tokenizer_wo_tf import EncoderUnitK


class BERTClient:
    def __init__(self, server_addr, port, seq_len=512):
        voca_path = os.path.join(data_path, "bert_voca.txt")
        self.encoder = EncoderUnitK(seq_len, voca_path)
        self.proxy = xmlrpc.client.ServerProxy('{}:{}'.format(server_addr, port))

    def request_single(self, text1, text2):
        payload = []
        payload.append(self.encoder.encode_pair(text1, text2))
        return self.send_payload(payload)

    def send_payload(self, payload):
        if payload:
            r = self.proxy.predict(payload)
            return r
        else:
            return []

    def request_multiple(self, text_pair_list: List[Tuple[str, str]]):
        payload = []
        for text1, text2 in text_pair_list:
            payload.append(self.encoder.encode_pair(text1, text2))
        return self.send_payload(payload)

    def request_multiple_from_tokens(self, payload_list: List[Tuple[List[str], List[str]]]):
        conv_payload_list = []
        for tokens_a, tokens_b in payload_list:
            conv_payload_list.append(self.encoder.encode_token_pairs(tokens_a, tokens_b))
        return self.send_payload(conv_payload_list)

    def request_multiple_from_ids(self, payload_list: List[Tuple[List[int], List[int]]]):
        conv_payload_list = []
        for tokens_a, tokens_b in payload_list:
            e = self.encoder.encode_inner(tokens_a, tokens_b)
            def flat(d):
                return d["input_ids"], d["input_mask"], d["segment_ids"]

            conv_payload_list.append(flat(e))
        return self.send_payload(conv_payload_list)

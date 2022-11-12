from typing import List, Optional, Tuple, Dict, NamedTuple

from main_code.data_structure.segmented_instance.seg_instance import SegmentedInstance
from main_code.data_structure.segmented_instance.segmented_text import SegmentedText
from main_code.ep_common import ReplaceSamplePolicyIF
from util.misc_lib import threshold05
from main_code.data_structure.eval_data_structure import RelatedEvalInstanceEx, RelatedBinaryAnswer
from main_code.metric_common.segment_modify_fn import get_replace_zero, get_partial_text_as_segment
from main_code.eval_metrics.eval_metric_common import EvalStateIF, RelatedMetric, StateWorkerIF, \
    FinalStateWorker, StateDone
from util.promise import MyFuture, MyPromise, list_future, PromiseKeeper


def add_w(text: SegmentedText, tokens_ids):
    n_last = len(text.tokens_ids)
    n_add = len(tokens_ids)
    new_seg_indices = list(range(n_last, n_last+n_add))
    new_indices = text.seg_token_indices + [new_seg_indices]
    return SegmentedText(text.tokens_ids + tokens_ids, new_indices)


class PerWord(NamedTuple):
    qt_w_dt_w: SegmentedInstance
    qt_w_w: SegmentedInstance
    word: List[int]



class SubstitutionStateBegin(EvalStateIF):
    def __init__(self,
                 problem: RelatedEvalInstanceEx,
                 answer: RelatedBinaryAnswer,
                 ):
        self.problem = problem
        self.answer = answer

    def get_code(self):
        return 0


class SubstitutionStateSearch(EvalStateIF):
    def __init__(self,
                 problem: RelatedEvalInstanceEx,
                 answer: RelatedBinaryAnswer,
                 word_list: List[PerWord],
                 ):
        self.problem = problem
        self.answer = answer
        self.word_list = word_list

    def get_code(self):
        return 1

    def get_pw(self) -> PerWord:
        return self.word_list[0]


class SubstitutionStateFound(EvalStateIF):
    def __init__(self,
                 problem: RelatedEvalInstanceEx,
                 answer: RelatedBinaryAnswer,
                 word: PerWord,
                 ):
        self.problem = problem
        self.answer = answer
        self.word = word

    def get_code(self):
        return 2


class SubstitutionStateDone(StateDone):
    def get_code(self):
        return 3


class SubstitutionStateWorkerBegin(StateWorkerIF):
    # 0 -> 1
    def __init__(self, get_word_pool):
        self.replace_zero = get_replace_zero()
        self.get_word_pool = get_word_pool

    def map(self, item: SubstitutionStateBegin) -> MyFuture[List[PerWord]]:
        assert item.get_code() == 0
        problem = item.problem
        answer = item.answer
        key = problem.problem_id
        word_pool: List[List[int]] = self.get_word_pool(key)
        if not word_pool:
            raise IndexError()
        pw_list: List[PerWord] = []
        qt = get_partial_text_as_segment(problem.seg_instance.text1, problem.target_seg_idx)
        for word in word_pool:
            qt_w = add_w(qt, word)
            dt_w = self.replace_zero(problem.seg_instance.text2, answer.score_table[problem.target_seg_idx], word)
            w = SegmentedText.from_tokens_ids(word)
            qt_w_dt_w: SegmentedInstance = SegmentedInstance(qt_w, dt_w)
            qt_w_w: SegmentedInstance = SegmentedInstance(qt_w, w)
            pw_list.append(PerWord(qt_w_dt_w, qt_w_w, word))

        future = MyFuture[List[PerWord]]()
        future.set_value(pw_list)
        return future

    def reduce(self, pw_list: MyFuture[List[PerWord]], item: SubstitutionStateBegin) -> EvalStateIF:
        return SubstitutionStateSearch(item.problem, item.answer, pw_list.get())


class SubstitutionStateWorkerSearch(StateWorkerIF):
    # 1 -> 2
    def __init__(self, pk):
        self.pk = pk

    def map(self, item: SubstitutionStateSearch) -> Optional[Tuple[MyFuture[float], MyFuture[float]]]:
        try:
            pw = item.get_pw()
            score_qt_w_dt_w_future = MyPromise(pw.qt_w_dt_w, self.pk).future()
            score_qt_w_w_future = MyPromise(pw.qt_w_w, self.pk).future()
            future_pair = score_qt_w_dt_w_future, score_qt_w_w_future
            assert item.get_code() == 1
            return future_pair
        except IndexError:
            return None

    def reduce(self, future_pair: Optional[Tuple[MyFuture[float], MyFuture[float]]],
               item: SubstitutionStateSearch) -> EvalStateIF:
        if future_pair is None:
            return SubstitutionStateDone(item.problem, item.answer, 1)
        else:
            score_qt_w_dt_w_future, score_qt_w_w_future = future_pair
            score_qt_w_dt_w = score_qt_w_dt_w_future.get()
            score_qt_w_w = score_qt_w_w_future.get()
            f1 = score_qt_w_dt_w >= 0.5
            f2 = score_qt_w_w >= 0.5
            satisfy: bool = f1 and not f2
            if satisfy:
                return SubstitutionStateFound(item.problem, item.answer, item.get_pw())
            else:
                return SubstitutionStateSearch(item.problem, item.answer, item.word_list[1:])


class SubstitutionStateWorkerEval(StateWorkerIF):
    # 2 -> 3
    def __init__(self, pk, replace_sample_policy):
        self.pk = pk
        self.replace_sample_policy = replace_sample_policy

    def map(self, item: SubstitutionStateFound) -> List[MyFuture[float]]:
        problem = item.problem
        answer = item.answer
        pw = item.word

        qt = get_partial_text_as_segment(problem.seg_instance.text1, problem.target_seg_idx)
        qt_w: SegmentedText = add_w(qt, pw.word)
        drop_doc_list: List[SegmentedText] = self.replace_sample_policy.get_replaced_docs(
            problem.seg_instance.text2,
            answer.score_table[problem.target_seg_idx],
            pw.word
        )
        si_list = [SegmentedInstance(qt_w, doc) for doc in drop_doc_list]
        future_predictions = [MyPromise(si, self.pk).future() for si in si_list]
        return future_predictions

    def reduce(self, future_predictions: List[MyFuture[float]],
               item: SubstitutionStateFound) -> SubstitutionStateDone:
        scores = list_future(future_predictions)
        score = self.replace_sample_policy.combine_results(scores)
        return SubstitutionStateDone(item.problem, item.answer, score)


class SubstitutionNess(RelatedMetric):
    def __init__(self, forward_fn,
                 replace_sample_policy: ReplaceSamplePolicyIF,
                 get_word_pool):
        pk = PromiseKeeper(forward_fn)
        self.pk = pk
        self.workers: Dict[int, StateWorkerIF] = {
            0: SubstitutionStateWorkerBegin(get_word_pool),
            1: SubstitutionStateWorkerSearch(pk),
            2: SubstitutionStateWorkerEval(pk, replace_sample_policy),
            3: FinalStateWorker(3)
        }

    def get_state_worker(self, state_code):
        return self.workers[state_code]

    def get_first_state(self, p: RelatedEvalInstanceEx, a: RelatedBinaryAnswer) -> EvalStateIF:
        return SubstitutionStateBegin(p, a)

    def do_duty(self):
        self.pk.do_duty(False, True)

    def is_final(self, code):
        return code == 3


class SubstitutionSuffStateDone(StateDone):
    def get_code(self):
        return 2


class SubstitutionSuffStateWorkerSearch(StateWorkerIF):
    # 1 -> 2
    def __init__(self, pk, discretize):
        self.pk = pk
        self.discretize = discretize

    def map(self, item: SubstitutionStateSearch) -> Optional[Tuple[MyFuture[float], MyFuture[float]]]:
        try:
            pw = item.get_pw()
            score_qt_w_dt_w_future = MyPromise(pw.qt_w_dt_w, self.pk).future()
            score_qt_w_w_future = MyPromise(pw.qt_w_w, self.pk).future()
            future_pair = score_qt_w_dt_w_future, score_qt_w_w_future
            assert item.get_code() == 1
            return future_pair
        except IndexError:
            return None

    def reduce(self, future_pair: Optional[Tuple[MyFuture[float], MyFuture[float]]],
               item: SubstitutionStateSearch) -> EvalStateIF:
        if future_pair is None:
            return SubstitutionSuffStateDone(item.problem, item.answer, 1)
        else:
            score_qt_w_dt_w_future, score_qt_w_w_future = future_pair
            score_qt_w_dt_w = score_qt_w_dt_w_future.get()
            score_qt_w_w = score_qt_w_w_future.get()
            f1 = score_qt_w_dt_w >= 0.5
            f2 = score_qt_w_w >= 0.5
            score = score_qt_w_dt_w
            if self.discretize:
                score = threshold05(score)
            if not f2:
                return SubstitutionSuffStateDone(item.problem, item.answer, score)
            else:
                return SubstitutionStateSearch(item.problem, item.answer, item.word_list[1:])


class SubstitutionSuff(RelatedMetric):
    def __init__(self, forward_fn,
                 discretize,
                 get_word_pool):
        pk = PromiseKeeper(forward_fn)
        self.pk = pk
        self.workers: Dict[int, StateWorkerIF] = {
            0: SubstitutionStateWorkerBegin(get_word_pool),
            1: SubstitutionSuffStateWorkerSearch(pk, discretize),
            2: FinalStateWorker(2)
        }

    def get_state_worker(self, state_code):
        return self.workers[state_code]

    def get_first_state(self, p: RelatedEvalInstanceEx, a: RelatedBinaryAnswer) -> EvalStateIF:
        return SubstitutionStateBegin(p, a)

    def do_duty(self):
        self.pk.do_duty(False, True)

    def is_final(self, code):
        return code == 2


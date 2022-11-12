import abc
from typing import List

from main_code.data_structure.eval_data_structure import RelatedEvalInstanceEx, RelatedBinaryAnswer
from main_code.eval_metrics.eval_metric_common import FinalStateWorker, \
    UnexpectedCodeException, StateWorkerIF, RelatedMetric, StateDone


class OneStepStateIF(abc.ABC):
    @abc.abstractmethod
    def get_code(self):
        pass


class OneStepStateBegin(OneStepStateIF):
    def __init__(self,
                 problem: RelatedEvalInstanceEx,
                 answer: RelatedBinaryAnswer,
                 ):
        self.problem = problem
        self.answer = answer

    def get_code(self):
        return 0


class OneStepStateDone(StateDone):
    def get_code(self):
        return 1


class SingleStateWorkerIF(StateWorkerIF):
    @abc.abstractmethod
    def do_duty(self):
        pass


class SingleStateEvalMetric(RelatedMetric):
    def __init__(self, begin_state_worker: SingleStateWorkerIF):
        self.begin_state_worker = begin_state_worker
        self.dummy_state_worker = FinalStateWorker(1)

    def get_first_state(self, p: RelatedEvalInstanceEx, a: RelatedBinaryAnswer):
        return OneStepStateBegin(p, a)

    def do_duty(self):
        self.begin_state_worker.do_duty()

    def get_state_worker(self, state_code):
        if state_code == 0:
            return self.begin_state_worker
        elif state_code == 1:
            return self.dummy_state_worker
        else:
            raise UnexpectedCodeException(state_code)

    def get_scores(self, state_list: List[OneStepStateDone]):
        return [s.get_score() for s in state_list]

    def is_final(self, code):
        return code == 1

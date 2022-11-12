from typing import List, Tuple
from typing import Optional

from main_code.data_structure.related_eval_instance import RelatedEvalInstance
from util.list_lib import right, left
from main_code.data_structure.eval_data_structure import RelatedBinaryAnswer, join_a_p, \
    RelatedEvalInstanceEx, PerProblemEvalResult
from main_code.eval_metrics.eval_metric_common import RelatedEvalMetricIF, EvalStateIF
from util.promise import parent_child_pattern


def unpack_problem(p: RelatedEvalInstance) -> List[RelatedEvalInstanceEx]:
    output: List[RelatedEvalInstanceEx] = []
    for i in range(p.seg_instance.text1.get_seg_len()):
        reix = RelatedEvalInstanceEx(p.problem_id, i, p.seg_instance, p.score)
        output.append(reix)
    return output


def run_eval_inner(a_p_list: List[Tuple[RelatedBinaryAnswer, RelatedEvalInstanceEx]],
                   eval_policy: RelatedEvalMetricIF
                   ) -> List[Optional[float]]:
    state_list: List[EvalStateIF] = [eval_policy.get_first_state(p, a) for a, p in a_p_list]
    done = False
    iter_idx = 0
    prev_n_remain = -1
    while not done:
        t_future_list = eval_policy.apply_map(state_list)
        eval_policy.do_duty()
        next_state_list: List[EvalStateIF] = eval_policy.apply_reduce(t_future_list, state_list)
        state_list = next_state_list
        done = all([eval_policy.is_final(s.get_code()) for s in state_list])
        n_remain = len([1 for s in state_list if not eval_policy.is_final(s.get_code())])
        do_print = n_remain != prev_n_remain
        if do_print:
            print()
            print("Iter {} , {} n_remain".format(iter_idx, n_remain), end="")
        else:
            print(".", end="")
        iter_idx += 1
    print()
    scores: List[Optional[float]] = eval_policy.get_scores(state_list)
    return scores


def run_eval(answer_list: List[RelatedBinaryAnswer],
             problem_list: List[RelatedEvalInstance],
             eval_policy: RelatedEvalMetricIF,
             ) -> List[PerProblemEvalResult]:
    a_p_list = join_a_p(answer_list, problem_list)
    Parent = Tuple[RelatedBinaryAnswer, RelatedEvalInstance]
    Child = Tuple[RelatedBinaryAnswer, RelatedEvalInstanceEx]

    def unpack_problem_ex(a_p: Tuple[RelatedBinaryAnswer, RelatedEvalInstance]) \
            -> List[Tuple[RelatedBinaryAnswer, RelatedEvalInstanceEx]]:
        a, p = a_p
        return [(a, c) for c in unpack_problem(p)]

    def work_for_children(children: List[Child]) -> List[Optional[float]]:
        return run_eval_inner(children, eval_policy)

    ltl: List[Tuple[Parent, List[Optional[float]]]] \
        = parent_child_pattern(a_p_list,
                               unpack_problem_ex,
                               work_for_children)
    problem_ids: List[str] = [p.problem_id for a, p in left(ltl)]
    return [PerProblemEvalResult(pid, res) for pid, res in zip(problem_ids, right(ltl))]

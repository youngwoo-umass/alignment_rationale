from typing import List
from main_code.data_structure.eval_data_structure import Alignment2D
from main_code.scorers.discretize import discretize_and_save
from main_code.scorers.exact_match_scorer import TokenExactMatchScorer
from main_code.scorers.random_score import RandomScorer
from main_code.data_structure.matrix_scorer_if import MatrixScorerIF
from main_code.data_structure.related_eval_instance import RelatedEvalInstance
from main_code.loader import load_mmde_problem
from main_code.related_answer_data_path_helper import save_related_eval_answer
from main_code.scorers.related_scoring_common import run_scoring


def get_method(method_name) -> MatrixScorerIF:
    if method_name == "random":
        scorer: MatrixScorerIF = RandomScorer()
    elif method_name == "exact_match":
        scorer: MatrixScorerIF = TokenExactMatchScorer()
    else:
        raise ValueError
    return scorer


def load_problem_run_scoring_and_save(method: str):
    print(f"load_problem_run_scoring_and_save(\"{method}\")")
    scorer: MatrixScorerIF = get_method(method)
    problems: List[RelatedEvalInstance] = load_mmde_problem()
    answers: List[Alignment2D] = run_scoring(problems, scorer)
    save_related_eval_answer(answers, method)


def main():
    for method in ["random", "exact_match"]:
        load_problem_run_scoring_and_save(method)
        discretize_and_save(method)


if __name__ == "__main__":
    main()

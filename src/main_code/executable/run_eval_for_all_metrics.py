from main_code.run_eval import run_eval_for_method_policy


def main():
    model_interface = "localhost"
    method_list = ["exact_match", "random"]

    metric_list = ["substitution_suff_soft",
                   "substitution_suff_binary",
                   "deletion_ness_soft",
                   "deletion_ness_binary",
                   "deletion_suff_soft",
                   "deletion_suff_binary",
                   "substitution_ness_soft",
                   "substitution_ness_binary",
                ]
    for metric in metric_list:
        for method in method_list:
            print(f"{method} - {metric}")
            run_eval_for_method_policy(method, metric, model_interface)


if __name__ == "__main__":
    main()

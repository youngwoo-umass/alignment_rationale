# Summary

Code for the SIGIR 2022 paper "[Alignment Rationale for Query-Document Relevance
](https://dl.acm.org/doi/abs/10.1145/3477495.3531883)"



# Requirements

To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

# Terms 

* 'related prediction' corresponds to the alignment explanation in the paper.
* 'metrics' are the procedures to evaluate how good each alignment explanation is.
* 'mmd' refers to the model trained on MSMARCO Document ranking dataset.


# Execution

run.sh file runs three python codes.
* main_code/executable/run_predict.py: It generates two alignment predictions (random and exact match)
* main_code/executable/run_eval_for_all_metrics.py: It evaluates above two predictions wtih 8 metrics. 
* main_code/executable/show_preference.py: It shows how frequently each of 8 metrics prefer random alignment or exact match alignment.

# Files

* data/align/problems.json: It contains the query-document pairs that are used for experiments.
  * It is representing texts as BERT tokenized IDs. 
  * data/align/problems_decoded.json is the decoded (plain text) version of problems.json.
  * Each of the "documents" are actually sentences that are extracted from relevant documents.
  * 'text1' is for query and 'text2' is for document.
* output/db/mmd_cache.sqlite: The model predictions that are used for the experiments are stored as cache. 
Thus, you don't need to run the actual BERT model to run the experiments. 
* model: It contains a BERT model fine-tuned for MSMARCO Document ranking.
  * The model can be run as separate process to response XMLRPC request from run_eval_for_all_metrics.py 
    * msmarco_model/run_mmd_server.py 
* output/related_scores/{method}.score: The alignment explanation scores (related prediction) made by each method. (json)
* output/binary_related_scores/{method}.score: The alignment explanations built by applying threshold to 'related_scores'. (json)
* output/binary_related_eval_score_all/{method}_{metric}.score: Scores that the metric evaluated on the method. (jsonl)



# Citation for this repository

Please cite our paper if you used the code in the paper:

```bibtex
@inproceedings{10.1145/3477495.3531883,
author = {Kim, Youngwoo and Rahimi, Razieh and Allan, James},
title = {Alignment Rationale for Query-Document Relevance},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531883},
doi = {10.1145/3477495.3531883},
abstract = {Deep neural networks are widely used for text pair classification tasks such as as adhoc information retrieval. These deep neural networks are not inherently interpretable and require additional efforts to get rationale behind their decisions. Existing explanation models are not yet capable of inducing alignments between the query terms and the document terms -- which part of the document rationales are responsible for which part of the query? In this paper, we study how the input perturbations can be used to infer or evaluate alignments between the query and document spans, which best explain the black-box ranker's relevance prediction. We use different perturbation strategies and accordingly propose a set of metrics to evaluate the faithfulness of alignment rationales to the model. Our experiments show that the defined metrics based on substitution-based perturbation are more successful in preferring higher-quality alignments, compared to the deletion-based metrics.},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2489–2494},
numpages = {6},
keywords = {neural network explanation, textual matching, document search, text alignment, query highlighting, token-level explanation},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```
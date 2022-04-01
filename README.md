# checking-hatecheck-code
This repository contains code for reproducing the experiments described in the paper:
"Checking HateCheck: a cross-functional analysis of behaviour-aware learning for hate speech detection",  accepted for publication at the [NLP Power! Workshop on Efficient Benchmarking in NLP](https://nlp-power.github.io/).


## Files
* hatecheck_finetuning.ipynb: fine-tunes models on HateCheck All split.
* leave1out_finetuning.ipynb: fine-tunes models on HateCheck FuncOut, IdentOut and ClassOut splits.
* target_task_data_gen.ipynb: prepares target task data for training and evaluation.
* target_task_eval.ipynb: evaluates models on target data.
* evaluation.ipynb: evaluates models on the HateCheck splits.
* error_analysis.ipynb: presents the samples with best and worst prediction changes after the fine-tuning procedure.

We use the HateCheck data made available by the authors in the following repo: https://github.com/paul-rottger/hatecheck-data.

# ClozeTesting

## Task Description

The task is aimed to predict the answers for the blank with the context of the blank, which can be formulated as a multi-choice classification problem. ClozeTest-all contains 930 candidate words for selection. 

## Attention weight analysis

The attention weight analysis is in the analysis_results folder.

## Dependency

- python 3.6 or 3.7
- torch==1.5.0
- transformers>=2.5.0


## Data

The data for cloze testing are collected from the the validation and test sets of CodeSearchNet. In this study we focused on analyzing JAVA programming language.  Each instance contains a masked code function, its docstring and the target word. 

 Data statistics of ClozeTest-all-Java are shown in the below table:
 

|       | # Examples |
|-------|------------|
| Train | 28,345     |
| Dev   | 6,075      |
| Test  | 6,075      |


## Run ClozeTest

You can run ClozeTest-all by the following command. It will automatically generate predictions to ` --output_dir`.

```shell
python code/run_cloze.py \
			--model microsoft/codebert-base-mlm \
			--cloze_mode all \
			--output_dir evaluator/predictions/
```

## Evaluator

We provide a script to evaluate predictions for ClozeTest-all, and report accuracy for the task. You can run by the following command:

```shell
python evaluator/evaluator.py \
			--answers evaluator/answers \
			--predictions evaluator/predictions
```

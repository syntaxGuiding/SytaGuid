# Code Translation

## Task Definition

In this study, given a piece of Java code, the task is to translate the code into C# version. Models are evaluated by BLEU scores, accuracy, and CodeBLEU scores.

## Attention weight analysis

The attention weight analysis is in the analysis_results folder.

## Dataset

The dataset is collected from CodeXGlue.


### Data Format

The dataset is in the "data" folder. Each line of the files is a function, and the suffix of the file indicates the programming language.

### Data Statistics

Data statistics of the dataset are shown in the below table:

|       | #Examples |
| ------- | :-------: |
|  Train  |   10,300  |
|  Valid  |      500   |
|   Test  |    1,000  |

### Dependency

- python 3.6 or 3.7
- torch==1.4.0
- transformers>=2.5.0
- pip install scikit-learn

### Fine-tune

```shell
cd code
$pretrained_model = the place where you download CodeBERT models e.g. microsoft/codebert-base
$output_dir = the place where you want to save the fine-tuned models and predictions
python run.py \
	--do_train \
	--do_eval \
	--model_type roberta \
	--model_name_or_path $pretrained_model \
	--config_name roberta-base \
	--tokenizer_name roberta-base \
	--train_filename ../data/train.java-cs.txt.java,../data/train.java-cs.txt.cs \
	--dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--learning_rate lr=5e-5 \
	--train_steps 100000 \
	--eval_steps 5000

```

### Inference

```shell
cd code
$output_dir = the place where you want to save the fine-tuned models and predictions
python run.py \
    	--do_test \
	--model_type roberta \
	--model_name_or_path roberta-base \
	--config_name roberta-base \
	--tokenizer_name roberta-base  \
	--load_model_path $output_dir/checkpoint-best-bleu/pytorch_model.bin \
	--dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
	--test_filename ../data/test.java-cs.txt.java,../data/test.java-cs.txt.cs \
	--output_dir $output_dir \
	--max_source_length 512 \
	--max_target_length 512 \
	--beam_size 5 \
	--eval_batch_size 16 
```

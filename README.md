# Beyond Self-learned Attention: Enhancing Transformer-based Models Using Attention Guidance 
### Companian website 

This repository contains the code, data and experiment results for the paper [Beyond Self-learned Attention: Enhancing Transformer-based Models Using Attention Guidance]

In this study we propose a novel syntax guided attention mechanism for Transformer-based models for software engineering tasks. The below figure shows learned self-attention heads attention weight assignment difference between default Transformer and our proposed attention guidance mechanism.


<img src="https://github.com/Jirigesi/4share/blob/main/figures/ag.png" width="480">

In this study, we conudcted experiments on three software engineering tasks: code clone detection, cloze test and code translation. The below table shows the details of our task description, dataset source, dataset size and the number of parameters in the Transformer model.

| Task | Dataset name | Language name Dataset size | Task defination |
|---|---|---| --- |
| Clone detection | BigCloneBentch | Jave | 900K/416K/416K| Predic whether function pairs are semanctic similar|
| Cloze test | CodeXGlue CT-all | Java | 28K/6K/6K | Predict the missing token in the code |
| Code translation | CodeXGlue CodeTrans| Java-C# | 10K/0.5K/1K | Translate the code from one language to another |

## Hyper-parameters and base pre-trained models  

#### Clone detection
  model_type=roberta <br>
  config_name=microsoft/codebert-base <br>
  model_name_or_path=microsoft/codebert-base <br>
  tokenizer_name=roberta-base <br>
  epoch=2 <br>
  block_size=400 <br>
  train_batch_size= 16 <br>
  eval_batch_size=32 <br>
  learning_rate=5e-5 <br>
  max_grad_norm=1.0 <br>

#### Cloze test
  model_type=roberta <br>
  config_name=microsoft/codebert-base-mlm <br>
  model_name_or_path=microsoft/codebert-base <br>
  tokenizer_name=roberta-base <br>
  epoch=2 <br>
  block_size=512 <br>
  train_batch_size= 16 <br>
  eval_batch_size=32 <br>
  learning_rate=5e-5 <br>
  max_grad_norm=1.0 <br>
  

#### Code translation
  model_type=roberta  <br>
	config_name=roberta-base <br>
	tokenizer_name=roberta-base <br>
	max_source_length=512 <br>
	max_target_length=512 <br>
	beam_size=5 <br>
	train_batch_size=16 <br>
	eval_batch_size=16 <br>
	learning_rate=5e-5 <br>
	train_steps=100000 <br>
	eval_steps=5000 <br>

## Experimental results

The experiment results for Clone detection are in the folder CloneDetection. 

The experiment results for Cloze test are in the folder ClozeTesting.

The experiment results for Code translation are in the folder CodeTranslation.


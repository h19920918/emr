# Episodic Memory Reader (EMR)

This work has been published on The Meeting of the Association for Computational
Linguistics (ACL 2019).  
Here is the paper of arxiv version in https://arxiv.org/abs/1903.06164.  
or you can find the paper of ACL anthology version in https://aclweb.org/anthology/P19-1434 

# Requirements
* Python 3.6.4  
* Pytorch >= 1.0.0  
* tensorboardX >= 1.6  
* tqdm  
* termcolor  
* boto3
* numpy
* (below is for only tvqa)
* pysrt
* h5py
* image
* seaborn
* matplotlib

# How to use

Recommend running with at least three GPUs.

## bAbI  
```shell
cd babi  
```

To pre-process and train the model,
```shell
python main-a3c.py \
  --task-id=[2, 22] \
  --prepro \
  --model=[FIFO, UNIFORM, LIFO, LRU_DNTM, R_EMR, T_EMR] \
  --log-dir=$DIR  
```

You can see other configuration in 'main-a3c.py'.  
If you have a pre-process, you do not have to do it again.  

## TriviaQA  

```shell
cd trivia  
```

Download TriviaQA dataset from http://nlp.cs.washington.edu/triviaqa  

To pre-process TriviaQA dataset,
```shell
python prepro.py  
```

To train the model,
```shell
python main-a3c.py \
  --task=[web, wikipedia] \
  --model=[FIFO, UNIFORM, LIFO, LRU_DNTM, R_EMR, T_EMR] \
  --rl-method=[a3c, policy, discrete] \
  --log-dir=$DIR  
```

You can see other configuration in 'main-a3c.py'.  

## Demo

```shell
python main-a3c.py --demo --ckpt=TRAINED_MODEL  
```

## Test

```shell
python main-a3c.py --test --ckpt=TRAINED_MODEL  
```

## TVQA
This code is mostly based on the original TVQA code (https://github.com/jayleicn/TVQA). 

In case of Transformer code, it is based on the BERT code (https://github.com/huggingface/pytorch-pretrained-BERT).

```shell
cd tvqa
```

### Dataset
To prepare the TVA dataset, please follow the instructions "Usage 1~3" of the original TVQA code (https://github.com/jayleicn/TVQA).

Codes for preprocessing are also included in this repository.

Preparing ImageNet feature is also needed.

In case of ImageNet feature, please divide it in to pieces using "sep_large.py".
 
### Usage:
0. Pretraining	
	```
	python main-a3c.py --pretrain \
		--log-dir=$PRE_DIR
	```
	
Pretraining the TVQA model is needed to get right results.
1. Training
    ```
    python main-a3c.py --input_streams sub imagenet \
		--no_ts \
		--pretrain-dir=$PRE_DIR \
		--model=[FIFO, UNIFORM, LIFO, LRU_DNTM, R_EMR, T_EMR] \
		--log-dir=$DIR
    ```

2. Test
    ```
    python main-a3c.py --test --ckpt=$DIR
    ```

3. Demo
	```
	python main-a3c.py --demo --ckpt=$DIR
	```

# Acknowledgements

This work was supported by Clova, NAVER Corp. and Institute for Information \&
communications Technology Promotion(IITP) grant funded by the Korea
government(MSIT) (No.2016-0-00563, Research on Adaptive Machine Learning
Technology Development for Intelligent Autonomous Digital Companion). 

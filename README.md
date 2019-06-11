# Episodic Memory Reader (EMR)

This work has been published on The Meeting of the Association for Computational
Linguistics (ACL 2019).  
Here is the paper of arxiv version in https://arxiv.org/abs/1903.06164.  

We are going to update other experiment (TVQA) as soon as possible.

# Requirements
* Python 3.6.4  
* Pytorch >= 1.0.0  
* tensorboardX >= 1.6  
* tqdm  
* termcolor  
* boto3  

# How to use

Recommend running with at least three GPUs.

## bAbI  
To pre-process and train the model,
```shell
python main-a3c.py --task-id=#NUMBER --prepro  --model=#NAME --log-dir=#DIR  
```

You can see other configuration in 'main-a3c.py'.  
If you have a pre-process, you do not have to do it again.  

## TriviaQA  
Download TriviaQA dataset from http://nlp.cs.washington.edu/triviaqa  

To pre-process TriviaQA dataset,
```shell
python prepro.py  
```

To train the model,
```shell
python main-a3c.py --task=#NAME --model=#NANE --rl-method=#NAME  
```

You can see other configuration in 'main-a3c.py'.  


## Demo and Test

```shell
python main-a3c.py --demo or test --ckpt=#TRAINED_MODEL  
```


# Acknowledgements

This work was supported by Clova, NAVER Corp. and Institute for Information \&
communications Technology Promotion(IITP) grant funded by the Korea
government(MSIT) (No.2016-0-00563, Research on Adaptive Machine Learning
Technology Development for Intelligent Autonomous Digital Companion). 

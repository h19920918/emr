# Episodic Memory Reader (EMR)

This work has been published on The Meeting of the Association for Computational
Linguistics (ACL 2019).  
Here is the paper of arxiv version in https://arxiv.org/abs/1903.06164.  

We are going to update other experiments (TVQA) as soon as possible.

# Requirements
Python 3.6.4  
Pytorch >= 1.0.0
tensorboardX >= 1.6  
tqdm  
termcolor  
boto3  

# How to use

[bAbI]  
'python main-a3c.py --task-id=#NUMBER --prepro  --model=#NAME --log-dir=#DIR'  

You can see other configuration in 'main-a3c.py'.  
If you have a pre-process, you do not have to do it again.

[TriviaQA]  
Download TriviaQA dataset from http://nlp.cs.washington.edu/triviaqa  

To pre-process, run 'python prepro.py'  
'python main-a3c --task=#NAME --model=#NANE --rl-method=#NAME'  

You can see other configuration in 'main-a3c.py'.  


# Acknowledgements

This work was supported by Clova, NAVER Corp. and Institute for Information \&
communications Technology Promotion(IITP) grant funded by the Korea
government(MSIT) (No.2016-0-00563, Research on Adaptive Machine Learning
Technology Development for Intelligent Autonomous Digital Companion). 

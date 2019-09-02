# TVQA
This code is mostly based on the original TVQA code (https://github.com/jayleicn/TVQA). 

In case of Transformer code, it is based on the BERT code (https://github.com/huggingface/pytorch-pretrained-BERT).

### Dataset
To prepare the TVA dataset, please follow the instructions "Usage 1~3" of the original TVQA code (https://github.com/jayleicn/TVQA).

Codes for preprocessing are also included in this repository.

Preparing ImageNet feature is also needed.

In case of ImageNet feature, please divide it in to pieces using "sep_large.py".

### Requirements:
- Python 3.6.4
- PyTorch >= 1.0
- tensorboardX
- pysrt
- tqdm
- h5py
- numpy
- matplotlib
- image
- seaborn
  
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

	To use Demo code, you need to download Frames data and set path of it.

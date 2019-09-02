# This code is used to process huge h5py file
# You need to run this before using
# imagenet features.

import h5py
import numpy as np
import torch
import os
from tqdm import tqdm
import time

vid_h5 = "./data/tvqa_imagenet_pool5_hq.h5"

print("Loading video...")
start_time = time.time()
vid_h5 = h5py.File(vid_h5, "r", driver="core")
print("FINISH -- Cost %s seconds" % (time.time() - start_time))

folder_name = "./data/imagenet/"
import pdb; pdb.set_trace()

for key in tqdm(vid_h5.keys()):
    with h5py.File(os.path.join(folder_name, key), 'w') as f:
        f.create_dataset('data', data=vid_h5[key])


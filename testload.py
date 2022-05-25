import time

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import pyrender
import trimesh
from numpy.linalg import inv
import pytorch_lightning as pl
import random

import time
import multiprocessing

# manager = multiprocessing.Manager()
# return_dict = manager.dict()
# jobs = []
# datalist = np.array([])
# start = time.time()
# def make_list(index):
#     datalist = np.array([])
#     for i in tqdm(range(index*10-10,index*10)):
#         with open('datasettest1/' + str(i) + '.npy', 'rb') as f:
#             a = np.load(f)
#             #print(a.shape)
#             datalist = np.append(datalist , a)
#     datalist = datalist.reshape(-1,5,128,128)
#     return datalist
    # return_dict[index] = datalist

# for i in range(1,5):
#     p = multiprocessing.Process(target=make_list, args=(i, return_dict))
#     jobs.append(p)
#     p.start()

# for proc in jobs:
#     proc.join()
# print(return_dict.values())
# datalist = np.array(return_dict.values())
# print("time", time.time() - start)
# #datalist = datalist.reshape(1000,-1)
# print(datalist.shape)

# for i in tqdm(range(1,6)):
#     datalist = make_list(i)
#     with open('datasettest2/' + str(i) + '.npy', 'wb') as f:
#         np.save(f, datalist)

# start = time.time()
with open('dataset/' + str(5) + '.npy', 'rb') as f:
    a = np.load(f)
    #a = a.reshape(4000,5,128,128)
    print(a.shape)
    for a_ in a[1:]:
        print(np.argwhere(a_))
    img = a[0].astype(np.uint8)
    #img1 = a[,0].astype(np.uint8)
    print(img.shape)
    cv2.imshow("img",img)
    cv2.waitKey(0)
#     cv2.imshow("img1",img1)
#     cv2.waitKey(0)
# # print("time", time.time() - start)
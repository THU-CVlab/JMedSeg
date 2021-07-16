import os
from numpy import load
from os.path import join as join

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
data = load('{}/pretrained_weights/R50+ViT-B_16.npz'.format(ROOT_DIR))
lst = data.files
for item in lst:
    print(item)
    # print(data[item])

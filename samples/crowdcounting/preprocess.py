# -*- coding: utf-8 -*-

import os
import sys
import cv2
import json
import numpy as np
from scipy.io import loadmat

import skimage.io
from skimage import transform, data
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath('../../')
sys.path.append(ROOT_DIR)

from mrcnn import visualize

# DATA_DIR = 'E:\\data\\dl\\tt\\'
DATA_DIR = 'E:/data/dl/jhu/'
SAVE_DIR = 'E:/data/dl/jhu_processed/'
TXT_DIR = 'E:/data/dl/jhu/'
# IMG_DIR = r'../../data/cc1.jpg'
# MAT_DIR = r'../../data/matdata/'
# NPY_DIR = r'../../data/npydata/'

# for dd,ss,ii in os.walk(MAT_DIR):
#     ll = MAT_DIR+ii[0]
#     print(ll, type(ll))
#     nn = loadmat(ll)


def generate_output_frommat(mat_path, mode, out_path):
    mat_pathi = mat_path + mode + '/gt/'
    _, _, name_dir = os.walk(mat_pathi)
    for mat_namei in name_dir:
        mat_namei = mat_pathi + mat_namei
        mati = loadmat(mat_namei)


def generate_output_fromtxt(txt_path, mode, out_path=''):
    txt_pathi = txt_path + mode + '/gt/'
    for txt_namei in os.listdir(txt_pathi):
        txt_namei = txt_pathi + txt_namei
        with open(txt_namei, encoding='utf-8') as txti:
            for line in txti:
                locs = line.split(' ')
                tmp = locs[:4]


def image_padding(shape, image):
    blank = np.zeros(shape)
    blank.fill(255)
    h, w, _ = image.shape
    # h = image.shape[0]
    # w = image.shape[1]
    blank[:h, :w, :] = image
    return blank


def jhu_label_process(file, save_path, ratio):
    labels = []
    with open(file, 'r', encoding='utf-8') as file_in:
        for line in file_in:
            line = line.split(' ')
            labeli = ''
            for i in line[:4]:
                labeli += str(int(int(i) * ratio)) + ' '
            labels.append(labeli)
    with open(save_path, 'w', encoding='utf-8') as file_out:
        for labeli in labels:
            file_out.write(labeli+'\n')


def image_scale_preprocess(img_path, mode):
    std = 1024
    size = (1024, 1024, 3)
    image_p = img_path + mode + '/images/'
    gt_p = img_path + mode + '/gt/'
    for img_name in os.listdir(image_p):
        name, _ = img_name.split('.')
        gt_name = name + '.txt'
        image = plt.imread(image_p+img_name)
        h, w, d = image.shape
        hr = std/h
        wr = std/w
        ratio = hr if hr < wr else wr
        # image = transform.resize(image, size)
        image = transform.rescale(image, scale=ratio)
        image = image_padding(size, image)
        plt.imsave(SAVE_DIR+mode+'/images/'+img_name, image)
        jhu_label_process(gt_p+gt_name, SAVE_DIR+mode+'/gt/'+gt_name, ratio)


# generate_output_fromtxt(TXT_DIR, mode='train')
image_scale_preprocess(DATA_DIR, 'val')

# IMAGE_DIR = os.path.join(ROOT_DIR, "data")
# iii = skimage.io.imread(DATA_DIR + 'val/images/2734.jpg')
#
# iii = transform.rescale(iii, 0.9)
# iii = image_padding((1024, 1024, 3), iii)
#
# plt.imsave('D:/padding2.jpg', iii)
# img[:h, :w, :] = ii

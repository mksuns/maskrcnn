# -*- coding: utf-8 -*-

import json
import os
import sys
import numpy as np
import skimage
import skimage.draw
import skimage.io

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Imass_ids = class_ids[_idx]
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

MAT_DIR = '/data2/data/jhu/jhu_crowd_v2.0/'
TXT_DIR = '/data2/data/fc/'
LOG_DIR = os.path.abspath('../../data/')
class_list = ['BG', 'person']


def get_counts(class_ids):
    counts = 0
    for i in class_ids:
        if i == 1:
            counts += 1
    return counts


def save_count(count):
    with open('../../logs/counts/count.txt') as file_in:
        file_in.write(count)


def yolov_label_transform(file_path):
    location = []
    with open(file_path, 'r', encoding='utf-8') as file_in:
        for line in file_in:
            line = line.split(' ')
            mid_x, mid_y, w, h = line[1:5]
            mid_x = 1920 * float(mid_x)
            mid_y = 1080 * float(mid_y)
            w = round(1920 * float(w))
            h = round(1080 * float(h))
            x_min = round(mid_x - (w / 2))
            y_min = round(mid_y - (h / 2))
            location.append([y_min, x_min, h, w])
            # print(y_min, x_min, h, w)
    return location


class CrowdConfig(Config):
    """configuration for training on the crowd_counting dataset"""
    NAME = 'crowd_count'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1  # background + object kinds
    IMAGE_MIN_DIM = 1080
    IMAGE_MAX_DIM = 1920
    # IMAGE_RESIZE_MODE = None
    DETECTION_MIN_CONFIDENCE = 0.6


class CountDataset(utils.Dataset):
    """generate crowd counting dataset"""

    def load_count_jhu(self, data_path, mode):
        self.add_class('count', 1, 'person')
        # class2id = {'person': 1}
        out_path = data_path + mode + '/gt/'

        for namei in os.listdir(out_path):
            loc = []
            class_id = []
            file_path = out_path + namei
            with open(file_path, encoding='utf-8') as file_in:
                for line in file_in:
                    loci = line.split(' ')
                    loci = loci[:4]
                    loc.append(loci)
                    class_id.append(1)

            namei = namei.split('.')
            namei = namei[0]
            img_path = data_path + mode + '/images/' + namei + '.jpg'
            image = skimage.io.imread(img_path)
            height, width = image.shape[:2]
            self.add_image('count', image_id=namei, path=img_path, class_id=class_id,
                           height=height, width=width, locs=loc)

    def load_count(self, data_path, mode):
        self.add_class('count', 1, 'person')
        # class2id = {'person': 1}
        out_path = data_path + mode + '/labels/'
        val_path = data_path + 'val' + '/labels/'	

        for namei in os.listdir(out_path):
            if mode == 'train':
                val_names = os.listdir(val_path)
                if namei in val_names:
                    continue
            file_path = out_path + namei
            loc = yolov_label_transform(file_path)
            class_id = [1] * len(loc)
            # with open(file_path, encoding='utf-8') as file_in:
            #     for line in file_in:
            #         loci = line.split(' ')
            #         loci = loci[:4]
            #         loc.append(loci)

            namei = namei.split('.')
            namei = namei[0]
            img_path = data_path + mode + '/images/' + namei + '.jpg'
            image = skimage.io.imread(img_path)
            height, width = image.shape[:2]
            self.add_image('count', image_id=namei, path=img_path, class_id=class_id,
                           height=height, width=width, locs=loc)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info['source'] != 'count':
            return super(self.__class__, self).load_mask(image_id)

        height = image_info['height']
        width = image_info['width']
        class_id = image_info['class_id']
        counts = len(class_id)
        mask = np.zeros([height, width, counts], dtype=np.uint8)
        for i, loc in enumerate(image_info['locs']):
            rr, cc = skimage.draw.rectangle((int(loc[0]), int(loc[1])), extent=(int(loc[2]), int(loc[3])))
            mask[rr, cc, i] = 1

        return mask.astype(np.bool), np.array(class_id, dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info['source'] == 'count':
            return info['path']

    def save_info(self, save_path):
        with open(save_path, 'w') as js:
            json.dump(self.image_info, js, indent=4)


def train():
    config = CrowdConfig()
    modeli = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

    # Training dataset
    dataset_train = CountDataset()
    dataset_train.load_count(TXT_DIR, 'train')
    dataset_train.prepare()
    dataset_train.save_info(LOG_DIR + '/fc_train_info.json')

    # Validation dataset
    dataset_val = CountDataset()
    dataset_val.load_count(TXT_DIR, 'val')
    dataset_val.prepare()
    dataset_val.save_info(LOG_DIR + '/fc_val_info.json')

    modeli.train(dataset_train, dataset_val, learning_rate=1e-5, epochs=5, layers='heads')


def number_predict(image):
    config = CrowdConfig()
    class_names = ['bg', 'person']
    modeli = modellib.MaskRCNN(mode='inference', config=config, model_dir=MODEL_DIR)
    weight_path = modeli.find_last()
    modeli.load_weights(weight_path, by_name=True)
    # image = skimage.io.imread(image)
    result = modeli.detect([image], verbose=1)
    r = result[0]
    counts = get_counts(r['class_ids'])
    # save_count(counts)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
    return counts


train()

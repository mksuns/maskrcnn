# -*- coding: utf-8 -*-

import json
import time
import os
import sys
import shutil
import pymysql
import numpy as np
import skimage
import skimage.draw
import skimage.io

begin = time.time()
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_count_0000.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

LABEL_DIR = 'E:/data/dl/fc/'
TXT_DIR = 'E:/data/dl/jhu/'
DATA_DIR = os.path.abspath('../../data/fc')
class_list = ['bg', 'head']


def get_counts(class_ids):
    counts = 0
    for i in class_ids:
        if i == 1:
            counts += 1
    return counts


def save_count(count):
    with open('../../logs/counts/count.txt') as file_in:
        file_in.write(count)


def delete_files(pathi):
    fileList = list(os.listdir(pathi))
    for file in fileList:
        t_path = pathi + '/' + file
        if os.path.isfile(t_path):
            os.remove(t_path)
        else:
            shutil.rmtree(t_path)


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
    NAME = 'head'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1  # background + object kinds
    IMAGE_MIN_DIM = 1080
    IMAGE_MAX_DIM = 1920
    # IMAGE_RESIZE_MODE = None
    DETECTION_MIN_CONFIDENCE = 0.6


class InferenceConfig(CrowdConfig):
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8


class CountDataset(utils.Dataset):
    """generate crowd counting dataset"""

    def load_count_jhu(self, data_path, mode):
        self.add_class('count', 1, 'head')
        # class2id = {'head': 1}
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
        self.add_class('head', 1, 'head')
        # class2id = {'head': 1}
        out_path = data_path + mode + '/labels/'

        for namei in os.listdir(out_path):
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
            self.add_image('head', image_id=namei, path=img_path, class_id=class_id,
                           height=height, width=width, locs=loc)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info['source'] != 'head':
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
        if info['source'] == 'head':
            return info['path']

    def save_info(self, save_path):
        with open(save_path, 'w') as js:
            json.dump(self.image_info, js, indent=4)


def train():
    config = CrowdConfig()
    modeli = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)
    modeli.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

    # Training dataset
    dataset_train = CountDataset()
    dataset_train.load_count(LABEL_DIR, 'train')
    dataset_train.prepare()
    # dataset_train.save_info(LOG_DIR + '/fc_train_info.json')

    # Validation dataset
    dataset_val = CountDataset()
    dataset_val.load_count(LABEL_DIR, 'train')
    dataset_val.prepare()
    # dataset_train.save_info(LOG_DIR + '/fc_val_info.json')

    modeli.train(dataset_train, dataset_val, learning_rate=1e-5, epochs=1, layers='heads')


def number_predict(modeli, image_path, save_path):
    # config = InferenceConfig()
    class_names = ['BG', 'head']
    # modeli = modellib.MaskRCNN(mode='inference', config=config, model_dir=MODEL_DIR)
    # weight_path = modeli.find_last()
    # modeli.load_weights(LOCAL_MODEL_PATH, by_name=True)

    image = skimage.io.imread(image_path)
    result = modeli.detect([image], verbose=1)
    r = result[0]
    counts = get_counts(r['class_ids'])
    # save_count(counts)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], path=save_path)
    return counts


def cycle_read():
    # 定时或循环从数据库中取图片数据，存到对应的文件夹
    timestamp = {26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0, 34: 0}
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', db='object_detect')
    cursor = conn.cursor()
    source_name = 'source_data'
    get_sql = "select * from {}".format(source_name)
    cursor.execute(get_sql)
    rows = cursor.fetchall()
    names = []
    for rowi in rows:
        img = rowi[1]
        camera_id = rowi[0]
        timei = rowi[2]
        if timei > timestamp[camera_id]:
            timestr = time.strftime('%Y%m%d%H%M%S', time.localtime())
            img_name = str(camera_id) + '-' + timestr + '.jpg'
            path = '../../detect_client/public/tmp_inputs/' + img_name
            names.append(img_name)
            file = open(path, 'w')
            file.write(img)
            file.close()
        else:
            raise ValueError("camera {} is not updated".format(camera_id))
    conn.close()
    return names


def cycle_get(names):
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', db='object_detect')
    cursor = conn.cursor()
    path = '../../detect_client/public/tmp_inputs/'
    for name in names:
        img_path = path + name
        img = open(img_path, 'r')


def run(interval):
    config = InferenceConfig()
    modeli = modellib.MaskRCNN(mode='inference', config=config, model_dir=MODEL_DIR)
    modeli.load_weights(LOCAL_MODEL_PATH, by_name=True)
    input_path = '../../detect_client/public/tmp_inputs/'
    output_path = '../../detect_client/public/tmp_outputs/'
    final_path = '../../detect_client/public/latest_outputs/'
    while True:
        time_s = time.strftime('%H%M', time.localtime())
        if time_s != '1700':
            try:
                # sleep for the remaining seconds of interval
                time_remaining = interval - time.time() % interval
                time.sleep(time_remaining)

                names = cycle_read()
                if len(names) == 9:
                    delete_files(output_path)
                    for namei in names:
                        number_predict(modeli, input_path+namei, output_path+namei)
                print("-" * 100)
            except Exception as e:
                print(e)


# configi = InferenceConfig()
# modeli = modellib.MaskRCNN(mode='inference', config=configi, model_dir=MODEL_DIR)
# modeli.load_weights(LOCAL_MODEL_PATH, by_name=True)

if __name__ == '__main__':
    intervali = 10
    while True:
        time_now = time.strftime('%H%M', time.localtime())
        if time_now == '0700':
            run(intervali)

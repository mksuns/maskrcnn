# -*- coding:utf-8 -*-

import os
import sys
import pymysql
from samples.crowdcounting.test import InferenceConfig, number_predict
import tensorflow as tf
from flask_cors import CORS
from flask import Flask, jsonify, request

graph = tf.get_default_graph()
# DEFAULT_DIR = os.path.abspath('../../detect_client/src/assets/')
INPUT_NAME = []
DAILY_NUM = []
ROOT_DIR = os.path.abspath("../../")
TEST_INPUT_DIR = os.path.join(ROOT_DIR, 'detect_client/public/test_inputs')
TEST_OUTPUT_DIR = os.path.join(ROOT_DIR, 'detect_client/public/test_outputs')
TMP_INPUT_DIR = os.path.join(ROOT_DIR, 'detect_client/public/tmp_inputs')
TMP_OUTPUT_DIR = os.path.join(ROOT_DIR, 'detect_client/public/tmp_outputs')
LATEST_OUTPUT_DIR = os.path.join(ROOT_DIR, 'detect_client/public/latest_outputs')
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_count_0000.h5")
# 项目的根目录起服务后，相对路径可能会改变
# basedir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(ROOT_DIR)
import mrcnn.model as modellib

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r'/*': {'origins': '*'}})

config = InferenceConfig()
model = modellib.MaskRCNN(mode='inference', config=config, model_dir=MODEL_DIR)
model.load_weights(LOCAL_MODEL_PATH, by_name=True)


def is_image(imagename):
    allowed = ['jpg', 'png', 'gif']
    name, ex = imagename.split('.')
    if ex in allowed:
        return True
    else:
        return False


def delete_file(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        os.remove(file_path)


@app.route('/', methods=['POST', 'GET'])
def head():
    return 'Welcome'


@app.route('/index', methods=['POST', 'GET'])
def index():
    # 展示各摄像头对应人数和当日变化曲线
    # camera_id = [26, 27, 28, 29, 30, 31, 32, 33, 34]
    id2name = {26: '', 27: '', 28: '', 29: '', 30: '', 31: '', 32: '', 33: '', 34: ''}
    daily_nums = {}
    camera_id = [26, 27]
    nums = {}
    result = {}
    i = 1
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', db='object_detect')
    cursor = conn.cursor()
    for idi in camera_id:
        daily_num = []
        sql = "select * from all_data where camera_id={} order by recognition_time".format(idi)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for rowi in rows:
            daily_num.append(rowi[5])
        i = len(daily_num)
        daily_nums[idi] = daily_num
        nums[idi] = daily_num[-1]
    indexs = [x+1 for x in range(i)]
    result['xdata'] = indexs
    result['nums'] = nums
    result['dailynums'] = daily_nums
    return jsonify(result)


@app.route('/detect', methods=['POST', 'GET'])
def detect():
    global INPUT_NAME, DAILY_NUM
    if request.method == 'POST':
        if len(INPUT_NAME) == 0:
            delete_file(TEST_INPUT_DIR)
        image = request.files['file']
        # image_name = image.filename
        if image and is_image(image.filename):
            image.save(os.path.join(TEST_INPUT_DIR, image.filename))
            INPUT_NAME.append(image.filename)
            # 检测图片并保存结果
        else:
            print('no image')

    elif request.method == 'GET':
        delete_file(TEST_OUTPUT_DIR)
        output_urls = {}
        with graph.as_default():
            for name in INPUT_NAME:
                img_url = os.path.join(TEST_INPUT_DIR, name)
                save_dir = os.path.join(TEST_OUTPUT_DIR, name)
                numi = number_predict(model, img_url, save_dir)
                output_urls[numi] = '/test_outputs/' + name
        INPUT_NAME = []
        return jsonify({'imgurls': output_urls})


@app.route('/show', methods=['GET', 'POST'])
def show_all():
    # 获取后端不断生成的检测图片及其对应人数
    # camera_id = [26, 27, 28, 29, 30, 31, 32, 33, 34]
    camera_id = [26, 27]
    result = {}
    nums = {}
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='123456', db='object_detect')
    cursor = conn.cursor()
    sql = "select * from all_data order by recognition_time desc"
    cursor.execute(sql)
    rows = cursor.fetchall()
    for i in range(2):
        rowi = rows[i]
        result[rowi[2]] = '/latest_outputs/' + rowi[4]
        nums[rowi[2]] = rowi[5]
    if len(nums) == 2:
        return jsonify({'urls': result, 'nums': nums})
    else:
        raise ValueError('some images is not updated')


if __name__ == "__main__":
    # app.debug = True
    app.run()

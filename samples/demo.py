import os
import sys
from time import time
import matplotlib.pyplot as plt
import skimage.io

start = time()
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from samples.crowdcounting.test import CrowdConfig

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# import coco


# %matplotlib inline
plt.show()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
LOCAL_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_count_0000.h5')
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "data")


def get_counts(class_ids):
    counts = 0
    for i in class_ids:
        if i == 1:
            counts += 1
    return counts


class InferenceConfig(CrowdConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # NAME = 'fc'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 1080
    IMAGE_MAX_DIM = 1920
    # NUM_CLASSES = 1 + 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(LOCAL_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
class_namei = ['BG', 'person']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
# image = []
# for i in file_names:
#     image.append(skimage.io.imread(os.path.join(IMAGE_DIR, i)))
image1 = skimage.io.imread(os.path.join(IMAGE_DIR, 'fc/fc_2601.jpg'))    # random.choice(file_names)
# image2 = skimage.io.imread(os.path.join(IMAGE_DIR, 'fc/fc_2601.jpg'))

# Run detection
results = model.detect([image1], verbose=1)

# print(results)

# Visualize results
r = results[0]
# print(r['rois'].shape, r['masks'].shape)
visualize.display_instances(image1, r['rois'], r['masks'], r['class_ids'],
                            class_namei, r['scores'])

print(get_counts(r['class_ids']))
end = time()
print(end - start)
# r1 = results[1]
# visualize.display_instances(image2, r1['rois'], r1['masks'], r1['class_ids'],
#                             class_names, r1['scores'])

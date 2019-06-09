# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime
# Root directory of the project
ROOT_DIR = '/home/cqiuac/maskrcnn_test/mask_rcnn/'

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco


# # Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
#
#
# # Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_coco.h5")
#用自己训练出来的权重
#MODEL_DIR = os.path.join(ROOT_DIR, "logs/shapes20190510T1035")
MODEL_DIR ='/home/cqiuac/maskrcnn_test/mask_rcnn/logs/shapes20190511T1800'
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_shapes_0020.h5")
# COCO_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

#IMAGE_DIR = os.path.join(ROOT_DIR, "train_data/pic/")
IMAGE_DIR ='/home/cqiuac/Mask_face/Mask_RCNN-master/temp'


# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")

# Directory of images to run detection on
# = os.path.join(ROOT_DIR, "images")

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 , 16 , 32 , 64 , 128 )  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
                   #exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'cgboyc','ekavaz','ggrego','jgloma','kaknig','mbutle','mdpove','QiuChuchu']
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
import skimage
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

a=datetime.now()
# Run detection
results = model.detect([image], verbose=1)
b=datetime.now()
# Visualize results
print("shijian",(b-a).seconds)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

# coding: utf-8

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import skimage.color
import skimage.io
import pandas as pd

from config import Config
import utils
import model as modellib
import visualize
from model import log

isTrain = False

ROOT_DIR = os.getcwd()

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class FishesConfig(Config):
    NAME = "fishes"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 1 + 15

    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  

    TRAIN_ROIS_PER_IMAGE = 32

    STEPS_PER_EPOCH = 100

    VALIDATION_STPES = 5
    
config = FishesConfig()
config.display()

class_names = [u'黑鲷',u'泥鱼',u'绿鳍马面鲀',u'花鲈',u'黑鲪',u'大泷六线鱼',u'条石斑',u'海星',u'海参',u'海鳗',u'水母',u'章鱼',u'细刺鱼',u'螃蟹',u'鲳鱼']

assert config.NUM_CLASSES == len(class_names)+1

if isTrain:
    class FishesDataset(utils.Dataset):

        def load_shapes(self, count, class_names, dataset_path):
            for i,class_name in enumerate(class_names):
                self.add_class("fishes", i+1, class_name)
            
            for i in range(count):
                image_path,mask_path,mask_annotations_path = self.random_image(dataset_path)
                self.add_image("fishes", image_id=i, path=image_path,
                               mask_path=mask_path,
                               mask_annotations_path=mask_annotations_path)
        
        def random_image(self,dataset_path):
            data_paths = glob.glob(dataset_path+'/'+'*')
            data_ids = [data_path.split('/')[-1] for data_path in data_paths]
            
            data_id = random.choice(data_ids)
            
            image_path = dataset_path+'/'+data_id+'/'+'dehaze_original.jpg'
            mask_path = glob.glob(dataset_path+'/'+data_id+'/'+'mask*.jpg')
            mask_annotations_path = dataset_path+'/'+data_id+'/'+'MaskAnnotations.csv'
            
            return image_path,mask_path,mask_annotations_path
        
        def load_mask(self,image_id):
            info = self.image_info[image_id]
            
            image = self.load_image(image_id)
            mask_path = info['mask_path']
            mask_annotations_path = info['mask_annotations_path']
            
            mask = np.zeros([image.shape[0],image.shape[1],len(mask_path)],dtype=np.uint8)
            for i,_mask_path in enumerate(mask_path):
                _mask = skimage.io.imread(_mask_path)
                _mask = skimage.color.rgb2gray(_mask)
                mask[:,:,i] = (_mask>0).astype(np.uint8)
                
            with open(mask_annotations_path,'rb') as f:
                df = pd.read_csv(f)
                class_ids = df.iloc[:,1].values
            
            return mask, class_ids.astype(np.int32)

    dataset_train = FishesDataset()
    dataset_train.load_shapes(1000, class_names, '/home/ubuntu/anaconda2/ZSCWork/detect1/dataset')
    dataset_train.prepare()

    dataset_val = FishesDataset()
    dataset_val.load_shapes(50, class_names, '/home/ubuntu/anaconda2/ZSCWork/detect1/dataset')
    dataset_val.prepare()

    image_ids = np.random.choice(dataset_train.image_ids, 4)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    init_with = "last"  

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        model.load_weights(model.find_last()[1], by_name=True)


    '''
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=1, 
                layers='heads')
    '''

    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=30, 
                layers="all")

class InferenceConfig(FishesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = model.find_last()[1]

assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

if isTrain:
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)
    
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    
    visualize.my_display_instances(original_image, gt_bbox[:,:4], gt_mask, gt_bbox[:,4], 
                                dataset_train.class_names, figsize=(8, 8))
    
    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.my_display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], name='test.png')
else:
    pass
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

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class FishesConfig(Config):
    NAME = "fishes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 14
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 100
    VALIDATION_STPES = 5

config = FishesConfig()
config.display()

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

import colorsys
def random_colors(N):
    hsv = [(0, 1, 1) for _ in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors
def parameter_init():
    Info_List = []
    Fish_Num = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    HaiXing_List = []
    return Info_List,Fish_Num,HaiXing_List,0
def compute_distants(info_list,Info_List):
    _distance_list = []
    for oy,ox,_,_,_,_ in info_list:
        _distance_list.append(list( map(lambda o:np.sqrt((o[0]-oy)**2+(o[1]-ox)**2),
                                                    Info_List) ))
    return np.array(_distance_list)

import copy
import gc
gc.enable()
import time
from nms import nms

from newautolevel import AutoLevel
from SetBox import SetBox
Dehaze = AutoLevel()
SetBox = SetBox()

class_names = [u'BG',u'黑鲷',u'泥鱼',u'绿鳍马面鲀',u'花鲈',u'黑鲪',u'大泷六线鱼',u'条石斑',u'海星',u'海参',u'海鳗',u'水母',u'章鱼',u'细刺鱼',u'螃蟹',u'鲳鱼']
images_path = os.path.join(ROOT_DIR,'testimages')
results_path = os.path.join(ROOT_DIR,'testresults')
if not os.path.exists(results_path):
    os.mkdir(results_path)

for image_path in glob.glob(os.path.join(images_path,'*')):
    result_path = os.path.join(results_path,image_path.split('\\')[-1])
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    
    ######################################################################################
    Info_List,Fish_Num,HaiXing_List,Start = parameter_init()
    ######################################################################################
    
    for count,_image_path in enumerate(sorted(glob.glob(os.path.join(image_path,'*.jpg')))):
        #_image_save_path = os.path.join(result_path,_image_path.split('\\')[-1][:-4])
        _image_save_path = result_path
        if not os.path.exists(_image_save_path):
            os.mkdir(_image_save_path)
        
        image = skimage.io.imread(_image_path)
        dehaze_image = Dehaze.autolevel(image)
        
        results = model.detect([dehaze_image], verbose=1)
        r = results[0]
        rois = copy.deepcopy(r['rois'])
        class_ids = copy.deepcopy(r['class_ids'])
        scores = copy.deepcopy(r['scores'])
        masks = copy.deepcopy(r['masks'])
        #########################################################################################
        info_list = []
        colors = []
        if rois.shape[0]>0:
            if Start==0:
                colors = random_colors(rois.shape[0])
                for i in range(rois.shape[0]):
                    y1,x1,y2,x2 = rois[i]
                    color = colors[i]
                    Info_List.append([ (y1+y2)/2.0,(x1+x2)/2.0,color,rois[i],class_ids[i],scores[i],masks[:,:,i] ])
                
                for i,fish_id in enumerate(class_ids):
                    Fish_Num[fish_id] = Fish_Num[fish_id] + 1
                Fish_Num[0] = sum(Fish_Num[1:])
                
                HaiXing_List.append(rois[class_ids==8])
                HaiXing_List.append(class_ids[class_ids==8])
                HaiXing_List.append(scores[class_ids==8])
                HaiXing_List.append(np.array(colors)[class_ids==8].tolist())
                HaiXing_List.append(masks[:,:,class_ids==8])
                try:
                    c = HaiXing_List[4].shape[2]
                except:
                    HaiXing_List[4] = HaiXing_List[4][:,:,np.newaxis]
                
                delete_row = np.where(class_ids==8)[0]
                rois = np.delete(rois,delete_row,axis=0)
                class_ids = np.delete(class_ids,delete_row,axis=0)
                scores = np.delete(scores,delete_row,axis=0)
                masks = np.delete(masks,delete_row,axis=2)
                del(Info_List[:])
                colors = random_colors(rois.shape[0])
                for i in range(rois.shape[0]):
                    y1,x1,y2,x2 = rois[i]
                    color = colors[i]
                    Info_List.append([ (y1+y2)/2.0,(x1+x2)/2.0,color,rois[i],class_ids[i],scores[i],masks[:,:,i] ])
                if Info_List:
                    Start += 1
                else:
                    continue
                if rois.shape[0]>=2:
                    keep = nms(np.concatenate((rois,scores.reshape(rois.shape[0],-1)),axis=1))
                    new_Info_List = []
                    for k in keep:
                        new_Info_List.append(Info_List[k])
                    Info_List = copy.deepcopy(new_Info_List)
            else:
                delete_row = np.where(class_ids==8)[0]
                rois = np.delete(rois,delete_row,axis=0)
                class_ids = np.delete(class_ids,delete_row,axis=0)
                scores = np.delete(scores,delete_row,axis=0)
                masks = np.delete(masks,delete_row,axis=2)
                
                for i in range(rois.shape[0]):
                    y1,x1,y2,x2 = rois[i]
                    info_list.append([ (y1+y2)/2.0,(x1+x2)/2.0,rois[i],class_ids[i],scores[i],masks[:,:,i] ])
                if info_list:
                    pass
                else:
                    continue
                
                distance_list = compute_distants(info_list,Info_List)
                
                Info_List_delete = []
                for i in range(distance_list.shape[0]):
                    if np.min(distance_list[i])==np.min(distance_list,axis=0)[np.argmin(distance_list[i])]:
                        colors.append(Info_List[np.argmin(distance_list[i])][2])
                        Info_List_delete.append(np.argmin(distance_list[i]))
                        info_list[i][3] = Info_List[np.argmin(distance_list[i])][4]
                    else:
                        colors.append(random_colors(1)[0])
                        Fish_Num[0] = Fish_Num[0] + 1
                        Fish_Num[info_list[i][3]] = Fish_Num[info_list[i][3]] + 1
                
                new_Info_List = []
                for i in range(len(Info_List)):
                    if i not in Info_List_delete:
                        if Info_List[i][3][0]<image.shape[0]-200 and Info_List[i][3][1]<image.shape[1]-300 and Info_List[i][3][2]>200 and Info_List[i][3][3]>300 and (Info_List[i][3][2]-Info_List[i][3][0])*(Info_List[i][3][3]-Info_List[i][3][1])<600000:
                            new_Info_List.append(Info_List[i])
                print('out num %s' % (len(new_Info_List)))
                
                assert len(info_list)==len(colors)
                for i in range(len(info_list)):
                    info_list[i].insert(2,colors[i])
                Info_List = copy.deepcopy(info_list)
                Info_List = Info_List + new_Info_List
                if len(Info_List)>=2:
                    _rois = []
                    _scores = []
                    for oy,ox,color,roi,class_id,score,mask in Info_List:
                        _rois.append(roi)
                        _scores.append(score)
                    _rois = np.array(_rois)
                    _scores = np.array(_scores)
                    keep = nms(np.concatenate((_rois,_scores.reshape(_rois.shape[0],-1)),axis=1))
                    _new_Info_List = []
                    for k in keep:
                        _new_Info_List.append(Info_List[k])
                    Info_List = copy.deepcopy(_new_Info_List)
                rois = []
                class_ids = []
                scores = []
                masks = Info_List[0][-1][:,:,np.newaxis]
                for oy,ox,color,roi,class_id,score,mask in Info_List:
                    rois.append(roi)
                    class_ids.append(class_id)
                    scores.append(score)
                    masks = np.concatenate((masks,mask[:,:,np.newaxis]),axis=2)
                masks = np.delete(masks,(0),axis=2)
                rois = np.array(rois)
                class_ids = np.array(class_ids)
                scores = np.array(scores)
        #####################################################################################################
                
                if new_Info_List:
                    for oy,ox,color,roi,class_id,score,mask in new_Info_List:
                        colors = colors + [color]
                
            rois = np.concatenate((rois,HaiXing_List[0]),axis=0)
            class_ids = np.concatenate((class_ids,HaiXing_List[1]),axis=0)
            scores = np.concatenate((scores,HaiXing_List[2]),axis=0)
            colors = colors + HaiXing_List[3]
            masks = np.concatenate((masks,HaiXing_List[4]),axis=2)
            
            if rois.shape[0]>=2:
                keep = nms(np.concatenate((rois,scores.reshape(rois.shape[0],-1)),axis=1))
                rois = rois[keep]
                class_ids = class_ids[keep]
                scores = scores[keep]
                masks = masks[:,:,keep]
                new_colors = []
                for k in keep:
                    new_colors.append(colors[k])
                colors = copy.deepcopy(new_colors)
            
            try:
                SetBox.getImg(image)
                SetBox.getMasks(masks)
                SetBox.getClass_Names(class_names)
                SetBox.getClass_Ids(class_ids)
                SetBox.setBox(colors=colors,fish_num=Fish_Num,
                    name=os.path.join(_image_save_path,_image_path.split('\\')[-1][:-4]+'_dehaze_Box.png'))
            except:
                pass
            
            plt.cla()
            plt.clf()
            plt.close('all')
            gc.collect()
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import time
import random
import colorsys
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

class SetBox():
    def __init__(self):
        pass
    
    def getImg(self,img):
        self.img = img
    
    def getMasks(self,masks):
        self.masks = masks
        
    def getClass_Names(self,class_names):
        self.class_names = class_names
    
    def getClass_Ids(self,class_ids):
        self.class_ids = class_ids
    
    def loadImgData(self,mask,value=0):
        try:
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        except:
            pass
        
        pos = np.where(mask>value)
        pos = list(zip(pos[0],pos[1]))
        
        return np.array(pos)
    
    def pca(self,data,feat=1):
        meanVals = np.mean(data, axis=0)
        DataAdjust = data - meanVals
        covMat = np.cov(DataAdjust, rowvar=0)
        eigVals,eigVects = np.linalg.eig(np.mat(covMat))
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[-feat]
        redEigVects = eigVects[:,eigValInd]
        lowDDataMat = DataAdjust * redEigVects
        reconMat = (lowDDataMat * redEigVects.T) + meanVals
        return lowDDataMat, np.array(reconMat)
    
    def getTwoPoints(self,reconMat):
        x = reconMat[:,1]
        y = reconMat[:,0]
        
        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)
        
        yx = np.array([[min_y,min_x],[max_y,min_x],[min_y,max_x],[max_y,max_x]],np.float64)
        delete_id = []
        
        for i,_yx in enumerate(yx):
            if np.any(np.all(_yx==reconMat,axis=1)):
                pass
            else:
                delete_id.append(i)
        
        yx = np.delete(yx,delete_id,axis=0)
        
        return yx
        
    def getFourPoints(self,mask):
        data = self.loadImgData(mask)
        
        lowDMat1, reconMat1 = self.pca(data,1)
        yx1 = self.getTwoPoints(reconMat1)
        yx1 = np.array(sorted(yx1.tolist(),key=lambda x:x[1],reverse=False),np.float64)
        
        lowDMat2, reconMat2 = self.pca(data,2)
        yx2 = self.getTwoPoints(reconMat2)
        yx2 = np.array(sorted(yx2.tolist(),key=lambda x:x[0],reverse=False),np.float64)
        
        #yx1_mean = np.average(yx1,axis=0)
        #yx2_shift = yx1-yx1_mean
        yx2_mean = np.average(yx2,axis=0)
        yx2_shift = yx1-yx2_mean
        
        FourPoints = np.concatenate((yx2+yx2_shift[0],yx2+yx2_shift[1]),axis=0)
        FourPoints[[2,3],:] = FourPoints[[3,2],:]
        
        FourPoints[:,[0,1]] = FourPoints[:,[1,0]]
        
        return FourPoints
    
    def getFourPointsList(self):
        FourPointsList = []
        for i in range(self.masks.shape[-1]):
            FourPointsList.append(self.getFourPoints(self.masks[:,:,i]))
        return FourPointsList
    
    def Rect2Box(self,Rect,scale=0.1):
        x01,y01 = (Rect[0][0]+Rect[1][0])/2.,(Rect[0][1]+Rect[1][1])/2.
        x23,y23 = (Rect[2][0]+Rect[3][0])/2.,(Rect[2][1]+Rect[3][1])/2.
        
        x0,y0 = x01-np.abs(scale*(Rect[2][0]-Rect[1][0])),y01-np.abs(scale*(Rect[2][1]-Rect[1][1]))
        x1,y1 = x01+np.abs(scale*(Rect[2][0]-Rect[1][0])),y01+np.abs(scale*(Rect[2][1]-Rect[1][1]))
        
        x2,y2 = x23-np.abs(scale*(Rect[2][0]-Rect[1][0])),y23-np.abs(scale*(Rect[2][1]-Rect[1][1]))
        x3,y3 = x23+np.abs(scale*(Rect[2][0]-Rect[1][0])),y23+np.abs(scale*(Rect[2][1]-Rect[1][1]))
        
        return [[x0,y0],Rect[1],[x1,y1],Rect[0]],[[x2,y2],Rect[2],[x3,y3],Rect[3]]
    
    def BoxPlusShift(self,rect1,rect2,x_shift=5,y_shift=10):
    
        rect1[0][0] = rect1[0][0]-x_shift
        rect1[0][1] = rect1[0][1]-y_shift
        rect2[0][0] = rect2[0][0]-x_shift
        rect2[0][1] = rect2[0][1]-y_shift
        
        rect1[2][0] = rect1[2][0]+x_shift
        rect1[2][1] = rect1[2][1]+y_shift
        rect2[2][0] = rect2[2][0]+x_shift
        rect2[2][1] = rect2[2][1]+y_shift
        
        return rect1,rect2

    def plotBox(self,ax,rect1,rect2,color,linewidth=1,linestyle="dashed"):
        '''
        def plotRect(ax,rect):
            for i,(x,y) in enumerate(rect):
                if (i+1)==len(rect):
                    i_ = 0
                else:
                    i_ = i+1
                ax.add_line(Line2D((x,rect[i_][0]),(y,rect[i_][1]),linewidth=linewidth,color=color,linestyle=linestyle))
            return ax
        
        ax = plotRect(ax,rect1)
        ax = plotRect(ax,rect2)
        '''
        '''
        ax.add_line(Line2D((rect1[0][0],rect1[1][0]),(rect1[0][1],rect1[1][1]),linewidth=linewidth,color=color))
        ax.add_line(Line2D((rect1[1][0],rect1[2][0]),(rect1[1][1],rect1[2][1]),linewidth=linewidth,color=color))
        ax.add_line(Line2D((rect1[2][0],rect1[3][0]),(rect1[2][1],rect1[3][1]),linewidth=linewidth,color=color))
        ax.add_line(Line2D((rect1[3][0],rect1[0][0]),(rect1[3][1],rect1[0][1]),linewidth=linewidth,color=color))
        
        ax.add_line(Line2D((rect2[0][0],rect2[1][0]),(rect2[0][1],rect2[1][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        ax.add_line(Line2D((rect2[1][0],rect2[2][0]),(rect2[1][1],rect2[2][1]),linewidth=linewidth,color=color))
        ax.add_line(Line2D((rect2[2][0],rect2[3][0]),(rect2[2][1],rect2[3][1]),linewidth=linewidth,color=color))
        ax.add_line(Line2D((rect2[3][0],rect2[0][0]),(rect2[3][1],rect2[0][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        
        ax.add_line(Line2D((rect1[1][0],rect2[1][0]),(rect1[1][1],rect2[1][1]),linewidth=linewidth,color=color))
        ax.add_line(Line2D((rect1[2][0],rect2[2][0]),(rect1[2][1],rect2[2][1]),linewidth=linewidth,color=color))
        ax.add_line(Line2D((rect1[3][0],rect2[3][0]),(rect1[3][1],rect2[3][1]),linewidth=linewidth,color=color))
        ax.add_line(Line2D((rect1[0][0],rect2[0][0]),(rect1[0][1],rect2[0][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        '''
        ax.add_line(Line2D((rect1[0][0],rect1[1][0]),(rect1[0][1],rect1[1][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        ax.add_line(Line2D((rect1[1][0],rect1[2][0]),(rect1[1][1],rect1[2][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        ax.add_line(Line2D((rect1[2][0],rect1[3][0]),(rect1[2][1],rect1[3][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        ax.add_line(Line2D((rect1[3][0],rect1[0][0]),(rect1[3][1],rect1[0][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        
        ax.add_line(Line2D((rect2[0][0],rect2[1][0]),(rect2[0][1],rect2[1][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        ax.add_line(Line2D((rect2[1][0],rect2[2][0]),(rect2[1][1],rect2[2][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        ax.add_line(Line2D((rect2[2][0],rect2[3][0]),(rect2[2][1],rect2[3][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        ax.add_line(Line2D((rect2[3][0],rect2[0][0]),(rect2[3][1],rect2[0][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        
        ax.add_line(Line2D((rect1[1][0],rect2[1][0]),(rect1[1][1],rect2[1][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        ax.add_line(Line2D((rect1[2][0],rect2[2][0]),(rect1[2][1],rect2[2][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        ax.add_line(Line2D((rect1[3][0],rect2[3][0]),(rect1[3][1],rect2[3][1]),linewidth=linewidth,color=color,linestyle=linestyle))
        ax.add_line(Line2D((rect1[0][0],rect2[0][0]),(rect1[0][1],rect2[0][1]),linewidth=linewidth,color=color,linestyle=linestyle))
    
        return ax

    def random_colors(self,N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors
    
    def setRect(self,figsize=(8,8),colors=None,fish_num=None,name='test_rect.png'):
        fig,ax = plt.subplots(1,figsize=figsize)
        height, width = self.img.shape[:2]
        ax.set_ylim(height + 20, -20)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ################################################################################
        '''
        class_count = []
        for i,class_name in enumerate(self.class_names):
            class_count.append(np.sum((self.class_ids==i).astype(np.uint8)))
        caption = '{}:{} '.format('ALL',np.sum(class_count))
        for i,(_name,count) in enumerate(zip(self.class_names,class_count)):
        '''
        caption = '{}:{} '.format('ALL',fish_num[0])
        for i,(_name,count) in enumerate(zip(self.class_names,fish_num)):
            if i>=1 and i!=5:
                caption = caption + '{_name}:{count} '.format(_name=_name,count=count)
            elif i==5:
                caption = caption + '\n{_name}:{count} '.format(_name=_name,count=count)
            else:
                pass
        #print(caption)
        ax.text(13, -50, caption,color='w', size=11, backgroundcolor='b')
        ################################################################################
        
        Rects = self.getFourPointsList()
        #colors = self.random_colors(len(Rects))
        for i,Rect in enumerate(Rects):
            for j,(x,y) in enumerate(Rect):
                if (j+1)==len(Rect):
                    j_ = 0
                else:
                    j_ = j+1
                ax.add_line(Line2D((x,Rect[j_][0]),(y,Rect[j_][1]),linewidth=3,color=colors[i],linestyle="-"))
            #label
            class_id = self.class_ids[i]
            label = self.class_names[class_id]
            caption = " {} {} ".format(label, '%.1f*%.1f' % (np.sqrt((Rect[0][0]-Rect[3][0])**2+(Rect[0][1]-Rect[3][1])**2),np.sqrt((Rect[0][0]-Rect[1][0])**2+(Rect[0][1]-Rect[1][1])**2)))
            X = np.array([Rect[3][0]-Rect[0][0],Rect[3][1]-Rect[0][1]])
            Y = np.array([1,0])
            if X[1]<=0:
                ax.text(Rect[0][0],Rect[0][1],caption,color='w',size=8,backgroundcolor="r",clip_on=True,
                        rotation=np.arccos(X.dot(Y)/(np.sqrt(X.dot(X))*np.sqrt(Y.dot(Y))))*360/2/np.pi)
            else:
                ax.text(Rect[0][0],Rect[0][1],caption,color='w',size=8,backgroundcolor="r",clip_on=True,
                        rotation=-np.arccos(X.dot(Y)/(np.sqrt(X.dot(X))*np.sqrt(Y.dot(Y))))*360/2/np.pi)
        ax.imshow(self.img)
        plt.savefig(name,bbox_inches='tight',dpi=200)
        
    def setBox(self,figsize=(8,8),scale=0.01,colors=None,fish_num=None,name='test_box.png'):
        fig,ax = plt.subplots(1,figsize=figsize)
        height, width = self.img.shape[:2]
        ax.set_ylim(height + 20, -20)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        
        ################################################################################
        '''
        class_count = []
        for i,class_name in enumerate(self.class_names):
            class_count.append(np.sum((self.class_ids==i).astype(np.uint8)))
        caption = '{}:{} '.format('ALL',np.sum(class_count))
        for i,(_name,count) in enumerate(zip(self.class_names,class_count)):
        '''
        caption = '{}:{} '.format('ALL',fish_num[0])
        for i,(_name,count) in enumerate(zip(self.class_names,fish_num)):
            if i>=1 and i!=7:
                caption = caption + '{_name}:{count} '.format(_name=_name,count=count)
            elif i==7:
                caption = caption + '\n{_name}:{count} '.format(_name=_name,count=count)
            else:
                pass
        #print(caption)
        ax.text(13, -50, caption,color='w', size=11, backgroundcolor='b')
        ################################################################################
        
        Rects = self.getFourPointsList()
        #colors = self.random_colors(len(Rects))
        for i,Rect in enumerate(Rects):
            rect1,rect2 = self.Rect2Box(Rect,scale=scale)
            x_shift=random.randint(10,20)
            y_shift=random.randint(10,20)
            rect1,rect2 = self.BoxPlusShift(rect1,rect2,x_shift=x_shift,y_shift=y_shift)
            ax = self.plotBox(ax,rect1,rect2,colors[i],linewidth=1,linestyle="-")
            #label
            class_id = self.class_ids[i]
            label = self.class_names[class_id]
            caption = " {} {} ".format(label, '%.1f*%.1f' % (np.sqrt((Rect[0][0]-Rect[3][0])**2+(Rect[0][1]-Rect[3][1])**2)/20,np.sqrt((Rect[0][0]-Rect[1][0])**2+(Rect[0][1]-Rect[1][1])**2)/20))
            X = np.array([Rect[3][0]-Rect[0][0],Rect[3][1]-Rect[0][1]])
            Y = np.array([1,0])
            if X[1]<=0:
                ax.text(Rect[0][0],Rect[0][1],caption,color='w',size=8,backgroundcolor="r",clip_on=True,
                        rotation=np.arccos(X.dot(Y)/(np.sqrt(X.dot(X))*np.sqrt(Y.dot(Y))))*360/2/np.pi)
            else:
                ax.text(Rect[0][0],Rect[0][1],caption,color='w',size=8,backgroundcolor="r",clip_on=True,
                        rotation=-np.arccos(X.dot(Y)/(np.sqrt(X.dot(X))*np.sqrt(Y.dot(Y))))*360/2/np.pi)
        ax.imshow(self.img)
        plt.savefig(name,bbox_inches='tight',dpi=200)

if __name__=='__main__':
    test = SetBox()
    test.getImg(cv2.imread('test.jpg'))
    
    mask1 = cv2.cvtColor(cv2.imread('mask1.jpg'),cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
    mask2 = cv2.cvtColor(cv2.imread('mask2.jpg'),cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
    masks = np.concatenate((mask1,mask2), axis = 2)
    test.getMasks(masks)
    
    test.getClass_Names(['All','Turbot'])
    
    test.getClass_Ids(np.array([1,1,1,1,1]))
    
    #test.setRect()
    test.setBox(colors=[(0.1,1,1),(0.2,1,1),(0.3,1,1),(0.4,1,1),(0.5,1,1)],fish_num=[5,5],name='1.png')
    plt.show()
    































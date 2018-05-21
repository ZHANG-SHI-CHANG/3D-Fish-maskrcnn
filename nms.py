import numpy as np

def nms(Boxes,IOU_Thresh=0.2):
#Boxes x1,y1,x2,y2,score
#IOU_Thresh 
    
    x1 = Boxes[:,0]
    y1 = Boxes[:,1]
    x2 = Boxes[:,2]
    y2 = Boxes[:,3]
    scores = Boxes[:,4]
    
    areas = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    
    keep=[]
    while order.size>0:
        i = order[0]
        keep.append(i)
        
        w = np.maximum(0.0,np.minimum(x2[i],x2[order[1:]])-np.maximum(x1[i],x1[order[1:]])+1)
        h = np.maximum(0.0,np.minimum(y2[i],y2[order[1:]])-np.maximum(y1[i],y1[order[1:]])+1)
        _areas = w*h
        #IOU = _areas/(areas[i]+areas[order[1:]]-_areas)
        IOU = _areas/np.minimum(areas[i],areas[order[1:]])
        
        ids = np.where(IOU<=IOU_Thresh)[0]
        order = order[ids+1]
    
    return keep

if __name__=='__main__':
    Boxes = np.array([
                     [204,102,358,250,0.5],
                     [257,118,380,250,0.7],
                     [280,135,400,250,0.6],
                     [255,118,360,235,0.7],
                     [1,10,2,20,0.1]])
    print(nms(Boxes,0.3))

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# from detect_utils import rescale_csv
# rescale_csv("dh_data/box.csv","dh_data/box2scale.csv","dh_data/images/")

def iou_height_width(box1,box2):
    minx = tf.minimum(box1[...,0],box2[...,0])
    miny = tf.minimum(box1[...,1],box2[...,1])
    intersection = tf.maximum(0,minx*miny)
    area1 = box1[...,0]*box1[...,1]
    area2 = box2[...,0]*box2[...,1]
    union = area1+area2-intersection
    iou_hw = tf.maximum(0,intersection/union)
    return iou_hw


def iou(box1,box2):
    """
    EXAMPLE : input
    a1 = [[25,34,38,40],[27,235,93,325]]
    a2 = [[25,34,38,40],[23,235,93,345]]
    """
#     tf.debugging.assert_less_equal(box1[...,0],box1[...,2])
#     tf.debugging.assert_less_equal(box2[...,0],box2[...,2])
#     tf.debugging.assert_less_equal(box1[...,1],box1[...,3])
#     tf.debugging.assert_less_equal(box2[...,1],box2[...,3])
    box1  = tf.cast(box1,tf.float32)
    box2  = tf.cast(box2,tf.float32)
    xA = tf.maximum(box1[...,0],box2[...,0])
    xB = tf.minimum(box1[...,2],box2[...,2])
    yA = tf.maximum(box1[...,1],box2[...,1])
    yB = tf.minimum(box1[...,3],box2[...,3])
    intersection = tf.maximum(0.0,xB-xA+1)*tf.maximum(0.0,yB-yA+1)
    box1_area = (box1[...,2]-box1[...,0]+1)*(box1[...,3]-box1[...,1]+1)
    box2_area = (box2[...,2]-box2[...,0]+1)*(box2[...,3]-box2[...,1]+1)
    union = box1_area+box2_area-intersection
    iou_ = intersection/union
    return iou_

def xmin_ymin_centre_wh(box_array):
    width  = box_array[...,2]-box_array[...,0]
    height = box_array[...,3]-box_array[...,1]
    x_c = (width)/2
    x_c = box_array[...,0]+x_c
    y_c = (height)/2
    y_c = box_array[...,1]+y_c
    return tf.transpose(tf.cast([x_c,y_c,width,height],tf.float32))

def xw_xmin_ymin(rev_box):
    x_lengths = rev_box[...,2]//2
    y_lengths = rev_box[...,3]//2
    xmin = rev_box[...,0]-x_lengths
    xmax = rev_box[...,0]+x_lengths
    ymin = rev_box[...,1]-y_lengths
    ymax = rev_box[...,1]+y_lengths
    out = [xmin,ymin,xmax,ymax]
    return tf.transpose(tf.cast(out,tf.float32))

def rescale_csv(csv_in,csv_out,img_dir):

    temps = pd.read_csv(csv_in)
    he = []
    wi = []
    sca_xmin = []
    sca_xmax = []
    sca_ymin = []
    sca_ymax = []
    for x in temps.iterrows():
        file = cv2.imread(img_dir+x[1]["image"])
        h,w,_ = file.shape
        he.append(h)
        wi.append(w)
        xmins = sca_xmin.append(x[1]["xmin"]/w)
        ymins = sca_ymin.append(x[1]["ymin"]/h)
        xmaxs = sca_xmax.append(x[1]["xmax"]/w)
        ymaxs = sca_ymax.append(x[1]["ymax"]/h)

    temps["height"] = he
    temps["width"] = wi
    temps["xmin"] = sca_xmin
    temps["xmax"] = sca_xmax
    temps["ymin"] = sca_ymin
    temps["ymax"] = sca_ymax 

    temps.to_csv(csv_out,index=False)
    
    
def prepare_csv(train_csv_path,val_csv_path,exclude_label):
    label_map = {}
    train_csv_file = pd.read_csv(train_csv_path)
    val_csv_file   = pd.read_csv(val_csv_path)
    # format train
    train_csv_file = train_csv_file[~train_csv_file["labels"].isin(exclude_label)]
    dummy_str=train_csv_file["labels"]
    reindex=pd.factorize(train_csv_file['labels'])[0]+1
    train_csv_file["labels"] = reindex
    dummy_index = train_csv_file["labels"]
    # format val
    val_csv_file   = val_csv_file[~val_csv_file["labels"].isin(exclude_label)]
    reindex        = pd.factorize(val_csv_file['labels'])[0]+1
    val_csv_file["labels"] = reindex

    for s_x,i_y in zip(dummy_str.tolist()[:],dummy_index.tolist()[:]):
        label_map[i_y-1] =s_x

    classes = len(list(label_map.keys()))
    return train_csv_file,val_csv_file,classes,label_map
    
def basic_NMS(single_box,boxes):
    for i,b in enumerate(boxes):
        out = iou(b,single_box[:4])
        simval = out.numpy()
        if simval < 1.0 and simval > 0.85:
            return 
        else:
            return single_box
        
        
        
def get_predictions(fileinp,model):
    im_tensor = tf.io.decode_png(tf.io.read_file(fileinp),channels=3)
    im_tensor = tf.image.resize_with_pad(im_tensor,464,464)
    copy_tensor = im_tensor.numpy().copy()
    im_tensor = tf.expand_dims(im_tensor,axis=0)/255.
    out = model.predict(im_tensor)
    out = out[0]
    return out, copy_tensor


def process_predictions(out,confidence=0.3):
    collects = []
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            collects.append([(i,j),out[i,j]])
    best_matches = []
    for w in range(len(collects)):
        grid,box = collects[w]
        box1,c1,box2,c2,cla = box[:4],box[4],box[5:9],box[9],box[10:]
        if c1>= confidence:
            best_matches.append([grid,box1,cla])
        if c2>= confidence:
            best_matches.append([grid,box2,cla])
    boxes = []
    classes = []
    scores = []
    for q in range(len(best_matches)):
        grid,coord,cla = best_matches[q]
        cla = tf.nn.softmax(cla).numpy()
        gx,gy = grid
        coord[0] = coord[0]+gx
        coord[1] = coord[1]+gy
        boxes.append(coord*464/out.shape[0])
        classes.append(cla)
        scores.append(cla[np.argmax(cla,axis=-1)])
    boxes = xw_xmin_ymin(np.array(boxes)).numpy().astype(np.int)
    return boxes,classes,scores



def get_output(raw,boxes,classes,scores,label_map):
    raw = raw.astype(np.uint8)
    raw = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
    final_pred = []
    for b,c,s in zip(boxes,classes,scores):
        pred_vector = b.tolist()+[np.argmax(c)]+[s]
        simbox = basic_NMS(pred_vector,boxes)
        final_pred.append(simbox)

    for p in final_pred:
        xmin,ymin,xmax,ymax = p[:4]
        label = label_map[p[4]]
        score = round(p[-1]*100,3)
        rects = cv2.rectangle(raw,(xmin,ymin),(xmax,ymax),[0,0,255],2)
        cv2.putText(img=rects,text=str(label),org=(xmin,ymin),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=[255,5,30],thickness=1)
        cv2.putText(img=rects,text=str(score),org=(xmin,ymin+10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.3,color=[0,0,0],thickness=1)
#     plt.figure(figsize=(12,10))
#     plt.imshow(cv2.cvtColor(rects,cv2.COLOR_BGR2RGB))
    return rects
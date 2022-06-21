import numpy as np
import cv2
import tensorflow as tf


def predict(image_str,preprocessor,model,size=256):
    image_array =tf.io.read_file(image_str)
    image_array = tf.image.decode_png(image_array,channels=3)
    image_array = tf.image.resize(image_array,(size,size))
    image_disp = tf.cast(image_array,tf.uint8).numpy()
    image_array = preprocessor(image_array)
    image_array = tf.expand_dims(image_array,axis=0)
    prediction = model.predict(image_array)[0] #bg=0 stomach=1 lb = 2 sb=3
    prediction = tf.argmax(prediction,axis=-1).numpy()
    return prediction,image_disp

def get_mask(prediction,mapping):
    prediction = np.tile(np.expand_dims(prediction,axis=-1),(1,1,3))
    stomach_mask = np.all(prediction==1,axis=-1)
    large_mask = np.all(prediction==2,axis=-1)
    small_mask = np.all(prediction==3,axis=-1)
    prediction[stomach_mask,:]=mapping["stomach"]
    prediction[large_mask,:]=mapping["large_bowel"]
    prediction[small_mask,:]=mapping["small_bowel"]
    prediction = np.asarray(prediction,np.uint8)
    return prediction

def get_legend(mapping,size):
    legend = np.zeros((16,size,3),np.uint8)
    scale=256//3
    legend[:,:scale*1]=[0,0,0]
    legend[:,scale*1:scale*2]=[0,0,0]
    legend[:,scale*2:scale*3]=[0,0,0]
    font = cv2.FONT_HERSHEY_COMPLEX
    legend = cv2.putText(legend,"Stomach",org=(9,10),fontFace = font,fontScale=0.4,color=mapping["stomach"])
    legend = cv2.putText(legend,"Large bowel",org=(80,10),fontFace = font,fontScale=0.4,color=mapping["large_bowel"])
    legend = cv2.putText(legend,"Small bowel",org=(172,10),fontFace = font,fontScale=0.4,color=mapping["small_bowel"])
    return legend


def compose_visualize(image_disp,mask,mapping,size):
    visualize = (image_disp*0.8)+(mask*0.2)
    visualize = visualize.astype(np.uint8)
    legends = get_legend(mapping,size)
    visualize = np.vstack((legends,visualize))
    return visualize
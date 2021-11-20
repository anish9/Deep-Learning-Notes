import os
import string
from glob import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def build_from_data(data_root):
    char_from_data = []
    for files in os.listdir(data_root):
        for sub in os.listdir(data_root+"/"+files):
            for t in sub:
                char_from_data.append(t)
    char_uniq = sorted(list(set(char_from_data)))
    return char_uniq

def build_labels(label=None):
    tar2idx = dict()
    idx2tar = dict()
    if label:
        LABELS = label
    else:
        alphabets = [j for j in string.ascii_letters]
        numbers   = [str(j) for j in range(10)]
        LABELS    = alphabets+numbers
        print(LABELS)
        LABELS.append(" ")
   
    for i,s in enumerate(LABELS):
        tar2idx[s] = i
        idx2tar[i] = s
    return LABELS,tar2idx,idx2tar

def string_enc(xin,mapper,max_length):
    op = []
    for c in xin:
        op.append(c)
    op = [mapper[s] for s in op]
    op = pad_sequences([op],maxlen=max_length,padding="post")
    return op[0].tolist()

def get_data(dirs,mapper,max_length=10):
    images      = glob(dirs)
    targets     = [images[t].split("/")[-1].split("_")[0] for t in range(len(images))] #custom based on ocr string pattern
    targets     = [string_enc(c,mapper,max_length) for c in targets]
    return images,targets


def process_file(train_source,train_target,img_height,img_width):
    with tf.device('/CPU:0'):
        image  = tf.io.read_file(train_source)
        image  = tf.image.decode_png(image,channels=1)
        image  = tf.image.resize(image,(img_height,img_width))/255.
        image  = tf.transpose(image, perm=[1, 0, 2])
    return {"image":image,"label":train_target}




def val_preprocess(image,img_height,img_width):
    image  = tf.io.read_file(image)
    image  = tf.image.decode_png(image,channels=1)
    image  = tf.image.resize(image,(img_height,img_width))
    image_inp  = tf.transpose(image, perm=[1, 0, 2])/255.
    return image,image_inp


def predict(file,predictor,img_height,img_width,max_length):
    vis_image,inp_image = val_preprocess(file,img_height,img_width)
    inp_image = tf.expand_dims(inp_image,axis=0)
    ctc_pred = predictor.predict(inp_image)
    results = tf.keras.backend.ctc_decode(ctc_pred, input_length=[(max_length*2)], greedy=True)[0][0]
    results = results.numpy()[0].tolist()
    return results,vis_image
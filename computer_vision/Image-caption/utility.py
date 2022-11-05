import tensorflow as tf
import numpy as np
import re
import os
import json
import string
from tqdm import tqdm
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (EarlyStopping,TensorBoard,
                                        LearningRateScheduler,
                                        ModelCheckpoint)


def duplicate_merger(data,flag_const=0):
    for i,e in enumerate(data):
        try:
            if e == data[i+1]:
                data[i]=flag_const
        except:
            pass
    data = [i for i in data if i!=flag_const]
    return data


def tokenizer_saver(max_vocab_size,seq_length=10):
    dummy_layer = layers.Input(shape=(1,))
    text_vectorizer = layers.TextVectorization(max_tokens=max_vocab_size,
                                               output_sequence_length=seq_length,
                                               output_mode="int",standardize=None)(dummy_layer)
    tokenizer_model = tf.keras.models.Model(dummy_layer,text_vectorizer)
    return tokenizer_model

def save_tokenizer(max_vocab_size,seq_length,save_name,caption_list):
    Tokenizer_Model = tokenizer_saver(max_vocab_size,seq_length)
    layer_for_conversion = Tokenizer_Model.layers[-1]
    layer_for_conversion.adapt(caption_list)
    Tokenizer_Model.save(save_name+"/")
    return layer_for_conversion
    
def restore_tokenizer(save_name):
    load_tokenizer = tf.keras.models.load_model(save_name)
    vect_layer = load_tokenizer.layers[-1]
    return vect_layer

def get_activation(x,act):
    if act=="relu":
        x = tf.nn.relu(x)
    if act=="gelu":
        x = tf.nn.gelu(x)
    if act=="selu":
        x = tf.nn.selu(x)
    if act=="lrelu":
        x = tf.nn.leaky_relu(x)
    return x



def preprocess_fashion(text,imid):
    text = str(text)
    characters = string.punctuation
    text = text.lower()
    text = text.strip(characters)
    textlist = text.split(".")
    textlist = textlist[:-1]
    textlist = [re.sub("[%s]" %re.escape(characters),"",j) for j in textlist]
    textlist = ".".join(textlist)
    textlist = "<sos> "+textlist+" <eos>"

    return textlist,imid


def preprocess_generic(text,imid):
    text = str(text)
    characters = string.punctuation
    text = text.lower()
    text = text.strip(characters)
    textlist = text.split(" ")
    textlist = textlist[:-1]
    textlist = [re.sub("[%s]" %re.escape(characters),"",j) for j in textlist]
    textlist =" ".join(textlist)
    textlist = re.sub(' +', ' ',textlist)
    textlist = "<sos> "+textlist+" <eos>"

    return textlist,imid

def fashion_dataset_getter(caption_file,image_dir,count=100):
    caption_file = json.load(open(caption_file,"r"))

    img_root = image_dir
    image_path = os.listdir(img_root)
    imagedir_path = []
    raw_caption = []
    for f in tqdm(image_path):
        try:
            caption = caption_file[f]
            img_path = img_root+f
            captext,imdirtext = preprocess_fashion(caption,img_path)
            raw_caption.append(captext)
            imagedir_path.append(imdirtext)
        except Exception as e:
            pass

    lengths = [len(t.split()) for t in raw_caption]
    seq_max_length = max(lengths)+1

    imagedir_path = imagedir_path[:count]
    raw_caption = raw_caption[:count]
    return imagedir_path,raw_caption,seq_max_length



def preprocess_text_flickr(text,imid):
    text = str(text)
    characters = string.punctuation
    text = text.lower()
    text = text.strip(characters)
    textlist = text.split()
    textlist = [re.sub("[%s]" %re.escape(characters),"",j) for j in textlist]
    
    return " ".join(textlist),imid

def flickr_dataset_getter(caption_file,image_dir,count=1000):
    caption_file = json.load(open(caption_file,"r"))

    img_root = image_dir
    image_path = os.listdir(img_root)[:count]
    # imagedir_path = []
    # raw_caption = []
    dict_mapping_img_caption = dict()


    for f in tqdm(image_path):
        try:
            caption = caption_file[f]
            streak_length = len(caption)
            captions_list = [caption[j] for j in [0,1]] #take first 3 samples of each file
            for text_ in captions_list:
                caption_processed,image_id = preprocess_text_flickr(text_,image_dir+f)
                dict_mapping_img_caption[image_id] = "<sos> "+caption_processed+" <eos>"
        except:
            pass
        
    imagedir_path =  sorted(list(dict_mapping_img_caption.keys()))
    raw_caption = [dict_mapping_img_caption[i] for i in imagedir_path]
    return imagedir_path[:],raw_caption[:]

        

def get_callbacks(monitor_metric,ckpt_save_name,early_stop_count=3,logdir="tf_logging"):
    es = EarlyStopping(monitor=monitor_metric,patience=early_stop_count,verbose=1)
    ckpt =ModelCheckpoint(ckpt_save_name,monitor=monitor_metric,save_best_only=True,save_weights_only=True)
    tensorboard = TensorBoard(logdir)
    return [es,ckpt,tensorboard]


def preprocess_image(imagefile,image_size=(288,288)):
    imagefile = tf.io.read_file(imagefile)
    imarray = tf.image.decode_jpeg(imagefile)
    imarray = tf.image.resize_with_pad(imarray,target_height=image_size[0],target_width=image_size[1])
    imarray = tf.expand_dims(imarray,axis=0)
    return imarray

def predict_caption(imarray,model,vectorizer,mapping,length):
    new_seq = []
    output_seq = " "
    STARTER,ENDER = "<sos>","<eos>"
    output_seq+=STARTER
    for i in range(length):
        current = output_seq
        decode_sent = vectorizer([current])[:,:-1]
        pred = model.predict((imarray,decode_sent),verbose=0)

        word = mapping[np.argmax(pred[0,i,:])]
        if word == ENDER or word == ".":
            break
        output_seq+=" "+word
        sample = output_seq.split()[1:]
        
    for i in range(0,len(sample),8):
        block = sample[i:i+8]+["\n"]
        new_seq.extend(block)
    response = " ".join(new_seq)+"."
    return response
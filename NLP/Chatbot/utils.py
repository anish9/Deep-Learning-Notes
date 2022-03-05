"""utility codes"""


import pickle
import string
import re
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

def dataset_prep(x,y,maxlen=25):
    x = pad_sequences(x,maxlen=maxlen,padding="post",truncating="post")
    y = pad_sequences(y,maxlen=maxlen+1,padding="post",truncating="post")
    return x,y


def pipeline(x,y):
    x = tf.cast(x,tf.int32)
    y = tf.cast(y,tf.int32)
    data = ({"encoder_input":x,"decoder_input":y[:-1]}),y[1:]
    return data



def preprocess_text(text,block=None):
    text = str(text)
    characters = string.punctuation
    text = text.lower()
    text = text.strip(characters)
    list_text = text.split()
    try:
#         ignore = re.findall("[a-z]+["+characters+"]+[0-9|a-z|()*+,-./:;<=>?@[\\]^_`]+",text)[0]
#         text =[i for i in list_text if i != ignore]
#         text= " ".join(text)
        text = re.sub('[%s]' %re.escape(characters),"",text)
    except:
        text = " ".join(list_text)
        text = re.sub('[%s]' %re.escape(characters),"",text)
    if block == "target":
        return "<sos> "+text+" <eos>"
    return text
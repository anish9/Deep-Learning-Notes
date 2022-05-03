import tensorflow as tf
import scipy
import numpy as np
import  matplotlib.pyplot as plt

class TroubleShooter():
    def __init__(self,model,feature_layer_name,preprocessor,**kwargs):
        self.model = model
        self.layer_id = feature_layer_name
        self.preprocessor = preprocessor
        
    def prepare_image(self,image_name,image_size):
        image = tf.keras.preprocessing.image.load_img(image_name,target_size=image_size)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = self.preprocessor(image)[tf.newaxis,:,:,:]
        return image
    
    def compute_cam(self,arr,compute_grad_class=None):
        size= arr.shape[1:3]
        self.model.layers[-1].activation=None
        grad_model = tf.keras.models.Model(self.model.input,
                                           (self.model.get_layer(self.layer_id).output,
                                            self.model.output))
        with tf.GradientTape() as tape:
            fmap,pred = grad_model(arr)
            if compute_grad_class:
                compute_grad_class = pred[:,compute_grad_class]
                get_gradient = tape.gradient(compute_grad_class,fmap)
            else:
                indice_pred = tf.argmax(pred,axis=-1)
                indice_pred_value = pred[:,indice_pred[0]]
                get_gradient = tape.gradient(indice_pred_value,fmap)
        get_gradient = tf.reduce_mean(get_gradient,axis=(0,1,2)) #convert gradient matrix to a vector
        intensity_map = fmap[0]@get_gradient[:,tf.newaxis]
        intensity_map = tf.keras.layers.ReLU()(tf.squeeze(intensity_map))
        intensity_map =scipy.ndimage.zoom(intensity_map,
                                          (size[0]/intensity_map.shape[0],
                                           size[1]/intensity_map.shape[0]))
        plt.imsave("intensity.jpg",intensity_map)
        
    def get_class_names(self,arr,decoder):
        out = self.model.predict(arr)
        top_pred_class_ids = tf.argsort(out,direction="DESCENDING")[0][:3]
        print(top_pred_class_ids)
        out = decoder(out,top=3)[0]
        return out
        
    @staticmethod
    def visualize(image_name,target_size):
        rgb =  tf.keras.preprocessing.image.load_img(image_name,target_size=target_size)
        rgb = tf.keras.preprocessing.image.img_to_array(rgb,dtype=np.float32)
        hmap = tf.image.decode_jpeg(tf.io.read_file("intensity.jpg"))
        hmap = tf.cast(hmap,tf.float32)
        out = (hmap*0.6)+(rgb*0.4)
        
        result = tf.concat((rgb,out),axis=1)
        result = tf.keras.preprocessing.image.array_to_img(result)
        result.save("result.jpg")
        return result



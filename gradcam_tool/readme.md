## Usage
* Refer **Example.ipynb** Notebook for wider custom use cases.


### Load the model presets:
```
file_id = "assets/Elephant.jpg"
model_id = tf.keras.applications.resnet_v2.ResNet50V2(weights="imagenet")
size     = (224,224)

preprocessor = tf.keras.applications.resnet_v2.preprocess_input
decoder =tf.keras.applications.resnet_v2.decode_predictions
layer_id = 'post_relu' #last conv layer name
```
### Initalize the Troubleshooter object:
```
troubleshoot = TroubleShooter(model_id,layer_id,preprocessor)
```

### Run model interpretations
```
img_array = troubleshoot.prepare_image(file_id,size)
pprint(troubleshoot.get_class_names(img_array,decoder=decoder))
troubleshoot.compute_cam(img_array)
troubleshoot.visualize(file_id,size)
```

### Result interpretation of different models:
<p align = "center">
<img src = "https://github.com/anish9/Deep-Learning-Notes/blob/main/gradcam_tool/assets/resulteff.jpg">
</p>
<p align = "center">
GradCAM - EfficientNetB0
</p>

<p align = "center">
<img src = "https://github.com/anish9/Deep-Learning-Notes/blob/main/gradcam_tool/assets/resultresnet.jpg">
</p>
<p align = "center">
GradCAM - ResNet50V2
</p>

<p align = "center">
<img src = "https://github.com/anish9/Deep-Learning-Notes/blob/main/gradcam_tool/assets/resultxcep.jpg">
</p>
<p align = "center">
GradCAM - Xception
</p>

#### Credits : Keras.io Examples

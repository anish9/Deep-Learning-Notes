# Usage
* Refer Example.ipynb for proper understanding based on custom use cases.

## Load the model substances:
```
file_id = "assets/Elephant.jpg"
model_id = tf.keras.applications.resnet_v2.ResNet50V2(weights="imagenet")
size     = (224,224)

preprocessor = tf.keras.applications.resnet_v2.preprocess_input
decoder =tf.keras.applications.resnet_v2.decode_predictions
layer_id = 'post_relu' #last conv layer name

```
## Initalize the Troubleshooter object
```
troubleshoot = TroubleShooter(model_id,layer_id,preprocessor)
```

## Run model interpretations
```
img_array = troubleshoot.prepare_image(file_id,size)
pprint(troubleshoot.get_class_names(img_array,decoder=decoder))
troubleshoot.compute_cam(img_array)
troubleshoot.visualize(file_id,size)
```

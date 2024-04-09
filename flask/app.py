from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras import Sequential
from keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow_hub as hub
import os

import jsonpickle

app = Flask(__name__, template_folder='template')
 
model_path = 'litemask_model.h5'
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

model_file = '.h5'

if os.path.exists(model_file):
    model=load_model(model_file)
else:
    vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(128,128,3))

    for layer in vgg19.layers:
        layer.trainable = False

    model = Sequential()
    model.add(vgg19)
    model.add(Flatten())
    model.add(Dense(2, activation = "sigmoid"))

@app.route('/', methods=['GET', 'POST'])
def classif_imag():
  
  if 'image' not in request.files:
        return render_template('index.html', prediction="No image uploaded")

  imagefile = request.files['image']

  if imagefile.filename == '':
        return render_template('index.html', prediction="No image selected")

  image_path = "./images/" + imagefile.filename
   
  plt.figure(figsize=(5,5))
  #processing img
  k_img = cv2.imread(image_path)
  sample_mask_img = cv2.resize(k_img,(128,128))
  sample_mask_img = np.reshape(sample_mask_img,[1,128,128,3])
  sample_mask_img = sample_mask_img/255.0
##############################################
      #classification img
  class_img=model.predict(sample_mask_img)
  if class_img[0][0]>.51:
    result='Mask'
    color=(0,255,0)
  else:
    result='No mask'
    color=(255,0,0)
   ##########################################
        #out put
  img = cv2.cvtColor(k_img, cv2.IMREAD_GRAYSCALE)
  faces = face_model.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4) #returns a list of (x,y,w,h) tuples

  out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image

  #plotting
  for (x,y,w,h) in faces:
      cv2.rectangle(out_img,(x,y),(x+w,y+h),color,1)
      cv2.putText(out_img, result, (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
  a = plt.imshow(out_img)
  print(class_img[0][0])
  return (result) 

# model = tf.keras.models.load_model(
#        (model_path),
#        custom_objects={'KerasLayer':hub.KerasLayer}
# )

# with keras.utils.custom_object_scope({"LayerScale": LayerScale}):
#      model = tf.keras.models.load_model(model_path)
# try this



 
@app.route('/test', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="No image uploaded")
        
        imagefile = request.files['image']
        
        if imagefile.filename == '':
            return render_template('index.html', prediction="No image selected")

        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        # Extract features from the image
        image_array = extract_features(image_path)

        # Predict the label
        prediction = model.predict(image_array)
        label = "Positive" if prediction[0][0] > 0.5 else "Negative"

        return render_template('index.html', prediction=label, image_file=imagefile.filename)

    return render_template('web/templets/index.html')


  
if __name__ == '__main__':
    app.run(port=5000, debug=True)

from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow_hub as hub
import os

app = Flask(__name__, template_folder='template')
model_path = 'litemask_model.h5'

def load_my_model():
    # Load the MobileNet V2 feature extractor from TensorFlow Hub
    feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3))
    
    # Create a new model with the feature extractor as the base
    inputs = Input(shape=(224, 224, 3))
    x = feature_extractor_layer(inputs)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    
    return model

def extract_features(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = tf.expand_dims(image_array, axis=0)
    
    return image_array

model = load_my_model()

@app.route('/', methods=['GET', 'POST'])
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


 
# @app.route('/', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return render_template('index.html', prediction="No image uploaded")
        
#         imagefile = request.files['image']
        
#         if imagefile.filename == '':
#             return render_template('index.html', prediction="No image selected")

#         image_path = "./images/" + imagefile.filename
#         imagefile.save(image_path)

#         # Load and preprocess the image
#         image = load_img(image_path, target_size=(224, 224))
#         image_array = img_to_array(image)
#         image_array = preprocess_input(image_array)
#         image_array = tf.expand_dims(image_array, axis=0)

#         # Predict the label
#         prediction = model.predict(image_array)
#         label = "Positive" if prediction[0][0] > 0.5 else "Negative"

#         return render_template('index.html', prediction=label, image_file=imagefile.filename)

#     return render_template('web/templets/index.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)

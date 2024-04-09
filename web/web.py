#%%
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import keras
#%%
app = Flask(__name__,template_folder='template')

# Load the trained model
model = load_model("C:/Users/Dorcas/Desktop/web/mask_model.h5")

# Define a function to predict mask or no mask
def predict_mask(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    if prediction < 0.5:
        return "Mask"
    else:
        return "No Mask"

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_path = "./uploads/" + image_file.filename
            image_file.save(image_path)
            result = predict_mask(image_path)
            return render_template("result.html", prediction=result, image_path=image_file.filename)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(port=8080, debug=True)

# %%

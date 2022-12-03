# X-RAY-COVID-DETECTION
import streamlit as st
import tensorflow as tf
@st.cache(allow_output_mutation=True)
 def load_model():
model-tf.keras.models.load_model("model.h5")
return model
model-load_model()
st.write( Covid-19 Prediction
 file-st.file uploader("UPLOAD THE XRAY YOU HAVE", type-["jpeg","png"])
 import cv2 1) From PIL import Image, ImageOps
import numpy as np
 st.set_option('deprecation.shoufileUploaderEncoding', false) 16 def import and predict(image_data,model):
size=(64,64) image-Imageops.fit(image data, size, Image.ANTIALIAS)
Ing-np.asarray(Image)
 ing reshape-ing[np.newaxis,...] 21 prediction-model.predict(ing reshape)
 return prediction 23 if file is None:
st.text("Please upload an image file")
else:
image-Image.open(file)
st.image(image,use_column_width-True)
prediction import_and_predict(image,model) 29 class nanes- Sorry You Might Have Covid", "No-Covid"]
string": "class_names[np.argax(prediction)] st.success(string)

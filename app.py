# Importing necessary libraries
import tensorflow as tf
import numpy as np
import streamlit as st
import cv2
from PIL import Image, ImageOps
import os
from tensorflow.keras.models import load_model as tfk__load_model

# Setting up Streamlit page configuration and layout
st.set_page_config(layout="wide", page_title="Potato disease detection")
st.header("Potato Disease Classification")
hide_st_style = """
<style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Function to preprocess and make predictions on the uploaded image
def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

# Load the pre-trained model for potato disease classification
path = os.path.dirname(__file__)
model = tfk__load_model('model.h5')
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Streamlit UI layout using tabs
about_tab, app_tab, contacts_tab = st.tabs(['About', 'App', 'Contact'])

# About Tab: Information about the application and dataset
img = Image.open(("farmer.jpg"))
about_tab.subheader("Farmers are facing a lot of economic losses every year due to various diseases in crops. For potatoes, there are two main types of diseases: Early Blight and Late Blight.")
about_tab.image(img, caption='Farmer suffering losses due to crop diseases', use_column_width=True)
about_tab.subheader("If a farmer can detect these diseases early and apply the appropriate treatment, it can prevent the economic losses.")
about_tab.write("A convolutional neural network has been trained on the Plant Village dataset with an accuracy of 98.05%.")
about_tab.write("You can find the link to the Plant Village dataset in the following link:")
about_tab.markdown("[Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)")
about_tab.write("Link to GitHub repository:")
about_tab.markdown("[GitHub Repo](https://github.com/AbdelrahmanSabriAly/Potato_disease_classification.git)")

# App Tab: Allows users to upload an image and get predictions for potato disease classification
file = app_tab.file_uploader("Please upload an image of a potato leaf", type=["jpg", "png", "jpeg", "bmp"])
if file is None:
    app_tab.text("Please upload an image file")
else:
    image = Image.open(file)
    app_tab.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    confidence = round(100 * (np.max(predictions[0])), 2)
    idx = np.argmax(predictions[0])
    if idx == 0:
        app_tab.info("This potato suffers from Early Blight")
        app_tab.warning(f"Confidence: {confidence}%")
    elif idx == 1:
        app_tab.info("This potato suffers from Late Blight")
        app_tab.warning(f"Confidence: {confidence}%")
    else:
        app_tab.success("This potato is Healthy")
        app_tab.warning(f"Confidence: {confidence}%")

# Contact Tab: Displays contact information of the developer
contacts_tab.subheader("Abdelrahman Sabri Aly")
contacts_tab.write("Email: aaly6995@gmail.com")
contacts_tab.write("Phone: +201010681318")
contacts_tab.markdown("[WhatsApp:](https://wa.me/+201010681318)")
contacts_tab.markdown("[LinkedIn](https://www.linkedin.com/in/abdelrahman-sabri)")

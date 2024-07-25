<<<<<<< HEAD
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
model_file_path = '/content/FaceExpressions/fine_tuned_face_expression_model.h5'
model = load_model(model_file_path)

# Define class labels based on your dataset
class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']  # Adjust this list based on your classes

# Streamlit app
st.title("Face Expression Recognition")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Display the result
    st.write(f"Prediction: {predicted_label}")

    st.bar_chart(predictions[0], labels=class_labels)
=======
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the model
model_file_path = '/content/FaceExpressions/fine_tuned_face_expression_model.h5'
model = load_model(model_file_path)

# Define class labels based on your dataset
class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']  # Adjust this list based on your classes

# Streamlit app
st.title("Face Expression Recognition")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_labels[predicted_class]

    # Display the result
    st.write(f"Prediction: {predicted_label}")

    st.bar_chart(predictions[0], labels=class_labels)
>>>>>>> 1c1550c (model)

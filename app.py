import os
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import base64

subject_name = "Facial Expression Recognition"
subject_code = "AI101"
professor_name = "Bhavik Gandhi"

model_file_path = 'face_expression_model.h5'

if not os.path.exists(model_file_path):
    st.error(f"Model file not found: {model_file_path}")
else:
    model = load_model(model_file_path)

    class_labels = ['Ahegao', 'Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    def create_download_link(df, filename):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

    st.title('Face Expression Recognition')
    st.write(f'**Subject Name**: {subject_name}')
    st.write(f'**Subject Code**: {subject_code}')
    st.write(f'**Professor**: {professor_name}')
    st.write('Upload an image to see the predicted emotion.')

    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        results = []
        for uploaded_file in uploaded_files:
            with st.spinner(f'Processing {uploaded_file.name}...'):
                try:
                    image = load_img(uploaded_file, target_size=(224, 224))
                    image_array = img_to_array(image)
                    image_array = np.expand_dims(image_array, axis=0)
                    image_array = preprocess_input(image_array)

                    prediction = model.predict(image_array)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    predicted_label = class_labels[predicted_class]

                    if max(prediction[0]) < threshold:
                        st.warning(
                            f"The model is unsure about the prediction for {uploaded_file.name}. Highest confidence: {max(prediction[0]):.2f}")
                    else:
                        st.write(f'Predicted Emotion for {uploaded_file.name}: {predicted_label}')

                    st.image(image, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
                    st.write("Class Probabilities:")
                    probabilities = prediction[0]
                    for label, prob in zip(class_labels, probabilities):
                        st.write(f"{label}: {prob:.2f}")

                    results.append({
                        'Filename': uploaded_file.name,
                        'Predicted Emotion': predicted_label,
                        **{label: prob for label, prob in zip(class_labels, probabilities)}
                    })
                except Exception as e:
                    st.error(f"An error occurred with {uploaded_file.name}: {e}")

        if results:
            df_results = pd.DataFrame(results)
            st.markdown(create_download_link(df_results, 'predictions.csv'), unsafe_allow_html=True)

    st.write("### Instructions")
    st.write("1. Upload image files (jpg, jpeg, or png).")
    st.write("2. The app will display the predicted emotion along with class probabilities for each image.")
    st.write("3. Adjust the confidence threshold slider if needed to filter predictions based on confidence level.")

    st.write("### How It Works")
    st.write(
        "The model uses a Convolutional Neural Network (CNN) trained for emotion recognition. It processes the uploaded image and predicts the emotion with associated probabilities.")

    st.write("### Transfer Learning with VGG by Unfreezing Layers")
    st.write("**Objective**: Fine-tune a VGG model for facial expression recognition by unfreezing some of its pre-trained layers.")
    st.write("**Details:**")
    st.write("1. **Pre-trained Model**: VGG16 or VGG19.")
    st.write("2. **Approach**:")
    st.write("   - Load a pre-trained VGG model.")
    st.write("   - Initially freeze the convolutional base to retain the learned features.")
    st.write("   - Unfreeze some of the deeper convolutional layers and re-train these layers along with the new classification layers.")
    st.write("   - This approach allows the model to adjust learned features to better fit the new task.")
    st.write("**Benefits:**")
    st.write("   - Provides a balance between retaining pre-learned features and adapting the model to new data.")
    st.write("   - Helps improve the model’s performance for specific tasks by fine-tuning it on the new dataset.")

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Function to load the pre-trained model
def load_trained_model(model_path):
    return load_model(model_path)

# Function to process the uploaded image
def process_image(uploaded_image):
    img = image.load_img(uploaded_image, target_size=(200, 200), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to make prediction based on the model
def predict_image(model, img_array):
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Function to display the result based on the prediction
def display_prediction(predicted_class):
    if predicted_class == 0:
        st.subheader("Prediction: Normal Brain")
    else:
        st.subheader("Prediction: Tumor Detected")

# Function to handle the Streamlit app logic
def main():
    # Load the pre-trained model
    model = load_trained_model(r"D:\downloads\brain_tumor_model.h5")

    # Set the title of the app
    st.title("Brain Tumor Detection")
    st.write("Upload an MRI image to classify it as 'Normal' or 'Tumor'.")

    # Create a file uploader for users to upload an image
    uploaded_image = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

    # Check if the user has uploaded an image
    if uploaded_image is not None:
        # Process the uploaded image
        img_array = process_image(uploaded_image)
        
        # Show the uploaded image in the app
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Add a predict button
        if st.button("Predict"):
            # Perform the prediction using the model
            predicted_class = predict_image(model, img_array)
            
            # Display the prediction result
            display_prediction(predicted_class)

    else:
        st.write("Upload an image to make a prediction!")

# Run the main function to start the app
if __name__ == "__main__":
    main()

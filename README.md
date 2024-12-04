# Brain Tumor Detection Using Deep Learning

## Overview
This repository contains the implementation of a deep learning model (NN) for detecting brain tumors from MRI images. The model is trained on a dataset hosted on Kaggle.

## Dataset
The dataset contains images of brain MRI scans labeled as "Normal" or "Tumor." These images are used for detecting potential brain tumors by image classification. The dataset is divided into two categories:

Normal Brain
Tumor Detected
Dataset Link: Brain MRI Images for Brain Tumor Detection Dataset

My Kaggle Code: Brain Tumor Detection NN

**Dataset Link**: [Brain MRI Images for Brain Tumor Detection Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

**My Kaggle Code**: [Brain Tumor Detection](https://www.kaggle.com/code/fatmaabdulfattah/brain-tumor-detect-using-dnn/notebook)

## Description
This project demonstrates how to train a deep learning model to classify brain tumors using MRI images. It includes:

1. **Model Building**: A Fully Connected Neural Network (NN) is built using TensorFlow/Keras. The model includes layers like Dense, ReLU, and Softmax for classification.
2. **Model Training**: The model is trained on the dataset, where the images are resized and normalized to the appropriate size for the network input.
3. **Model Evaluation**: The model is evaluated using validation data to assess performance metrics like accuracy.
4. **Model Deployment**: After training and evaluation, the model is saved to a file (e.g., brain_tumor_model.h5) for use in predictions within the Streamlit app.
5. **Prediction Process**: Users can upload MRI images to the Streamlit app, which uses the trained model to predict if the brain is normal or contains a tumor.
6. **Result Display**: Once a prediction is made, the result is displayed along with the prediction confidence

## How the Streamlit App Works:
1. **Upload an Image**: Upload a clear image of the brain MRI scan. The app accepts JPG, PNG, or JPEG formats.
2. **Click Predict**: After the image is uploaded, click the 'Predict' button. The model will process the image and predict whether it's a "Normal" or "Tumor Detected" image.
3. **Receive Results**: The app will display the prediction result and confidence level. If a tumor is detected, users are advised to consult a doctor for further evaluation.


## Requirements
To run the notebook and Streamlit app, you need to install the following libraries:

### Kaggle Notebook Dependencies
- `tensorflow==2.14.0`
- `numpy==1.24.3`
- `opencv-python`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `pandas`

### Streamlit App Dependencies
- `streamlit==1.34.0`
- `pillow`



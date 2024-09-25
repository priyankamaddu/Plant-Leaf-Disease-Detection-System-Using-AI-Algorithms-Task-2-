# Plant Disease Detection System 🌿🔍

This project implements a Plant Disease Detection system using a Convolutional Neural Network (CNN) and TensorFlow. The system helps farmers and agriculturists by detecting plant diseases from images of crop leaves. The model is trained on a subset of the Plant Village dataset and uses a Streamlit web application for easy interaction with users.

## Key Features

- **CNN Model:** A deep learning model built with several convolutional layers to extract features from plant leaf images.
- **Image Preprocessing:** Images are resized to 128x128 pixels, converted to RGB, and fed into the model in batches.
- **Streamlit Integration:** A user-friendly interface for uploading plant images and predicting disease types.
- **Background Image:** Custom background image for a more aesthetic and engaging UI.

## How It Works

1. **Upload Image:** Users can upload an image of a plant leaf through the web interface.
2. **Disease Detection:** The CNN model processes the image and predicts the plant disease.
3. **Result:** The model outputs the predicted class of the disease, along with a display of the uploaded image.

## Dataset

- The dataset used in this project is from the [Plant Village dataset](https://www.kaggle.com/datasets/emmarex/plantdisease), containing over 26,000 images of healthy and diseased plant leaves categorized into 11 classes. 
- Data is split into training and validation datasets for effective model training and evaluation.
  - **Training:** 20,815 images
  - **Validation:** 5,203 images
  - **Test:** 14 images

## Model Architecture

The model is composed of the following layers:
- **Convolution Layers:** Extract features from images.
- **MaxPooling Layers:** Reduce the dimensionality of the feature maps.
- **Dropout Layers:** Prevent overfitting by randomly dropping some connections.
- **Dense Layer:** Fully connected layer for final classification of 38 plant disease categories.

### Layer Summary:
- 5 Convolutional Blocks with increasing filter sizes (32, 64, 128, 256, and 512).
- MaxPooling after each block.
- Dropout to regularize the network.
- Dense layer with 1240 units followed by the output layer with 38 units (softmax for multi-class classification).

## Installation

To run this project, you'll need to install the following dependencies:

```bash
pip install tensorflow numpy pandas matplotlib seaborn streamlit

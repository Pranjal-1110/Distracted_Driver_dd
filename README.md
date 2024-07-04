# Distracted Driver Detection
This repository contains the implementation and analysis of driver's behavioral pattern recognition using Convolutional Neural Networks (CNN) based on the dataset provided by State Farm on Kaggle.

## Project Overview
Detected drivers' behavioral patterns from a dataset consisting of over 20,000 images.
Developed a robust data pipeline to load, preprocess, and normalize images using OpenCV and NumPy.
Designed and implemented a CNN in Keras with 3 Conv2D layers and 2 Dense layers to classify driver behavior.
## Model Development
Designed a Convolutional Neural Network (CNN) architecture in Keras.
Consists of 3 Conv2D layers followed by max pooling layers for feature extraction.
Utilized 2 Dense layers for classification.
Trained the model with a total of 3,313,546 parameters, achieving a validation accuracy of 94% through rigorous parameter tuning.
## Techniques Used
Implemented dropout of 80% and batch processing techniques to enhance model training and prevent overfitting.
Employed data augmentation strategies to increase the diversity of training examples and improve model generalization.
## Results
Successfully applied the trained model to predict drivers' behavior in almost 80,000 images from the test dataset with high accuracy.
Evaluated model performance using metrics such as accuracy, precision, recall, and F1-score.
## Data Source
<a href = "https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data?select=imgs"> State Farm Distracted Driver Detection </a>

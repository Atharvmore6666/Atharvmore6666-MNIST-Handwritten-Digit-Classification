# Atharvmore6666-MNIST-Handwritten-Digit-Classification
This project demonstrates the application of a neural network to classify handwritten digits (0–9) using the MNIST dataset. The model was implemented using TensorFlow and Keras, achieving an impressive test accuracy of 92.94%. This project showcases the power of deep learning for image classification tasks.

## Motivation
The MNIST dataset serves as a classic benchmark for image recognition and machine learning projects. The objectives of this project include:
Developing a simple yet effective neural network for digit classification.
Learning and applying neural network design and evaluation techniques.
Visualizing model performance to interpret predictions and misclassifications.

## Dataset
The MNIST dataset used is a CSV file containing:
Pixel Values (784 columns): Flattened 28x28 grayscale images.
Labels (1 column): Ground truth digit labels ranging from 0 to 9.
## Dataset Splits
Training Set: 80% of the data for training and validation.
Test Set: 20% of the data for evaluation.
# Project Workflow
## Data Preparation:
Splitting the dataset into training and testing sets.
Visualizing sample images using Matplotlib.
## Neural Network Design:
Sequential model architecture with three dense hidden layers and one output layer.
Activation functions: ReLU (hidden layers) and Softmax (output layer).
## Model Compilation:
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metric: Accuracy
Training and Evaluation:
Training for 15 epochs with a batch size of 500 and a validation split of 20%.
Evaluating the model on the test set.
Performance Analysis:
Calculating accuracy and confusion matrix.
Visualizing the confusion matrix with Seaborn heatmaps.
Results
## Test Accuracy: 92.94%
## Confusion Matrix: 
Highlights minimal misclassifications, demonstrating strong performance across all digit classes.
## Conclusion
This project highlights the capability of deep learning to perform image classification with high accuracy. While the simple dense architecture performs well, further exploration using Convolutional Neural Networks (CNNs) or data augmentation can potentially enhance the results.




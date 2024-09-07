# MNIST CNN Image Classification

This repository provides a tutorial for building a custom Convolutional Neural Network (CNN) for image classification using the MNIST dataset. The model achieves an impressive **accuracy of 99.19%** on the test set. Additionally, this project visualizes the classification performance with a **confusion matrix**.

## Project Overview

The MNIST dataset consists of grayscale images of handwritten digits (0-9). The goal of this project is to classify these digits using a custom CNN.

### Key Features
- **Custom CNN**: The network is built from scratch, using `keras Sequential class` focusing on the fundamentals of CNN architecture.
- **High Accuracy**: The model achieves a **99.19% accuracy** on the test set.
- **Confusion Matrix**: A confusion matrix is plotted to visualize the model's classification performance across different digits.

## Dataset

The MNIST dataset is automatically downloaded through `tensorflow` or `keras` when running the code, so no manual download is necessary.

## Model Architecture

The CNN architecture is designed as follows:

- **Input Layer**: 28x28 grayscale images (flattened to 1 channel).
- **Convolutional Layers**: Multiple convolutional layers to capture spatial features.
- **Pooling Layers**: Max pooling for downsampling.
- **Dropout Layer**: Dropout layer to prevent overfitting by randomly setting a fraction of input units to 0 during training.
- **Fully Connected Layer**: Dense layer for classification.
- **Output Layer**: 10 neurons corresponding to the 10 digit classes (0-9).

## Training

To train the model, simply run the `MNIST_CNN.ipynb` notebook in jupyter server or google colab.\
The model will be trained on the MNIST training dataset and evaluated on the test dataset. The final test accuracy will be displayed, along with a confusion matrix.

## Results

- **Accuracy**: The model achieves an accuracy score of **99.19%** on the test set.
- **Confusion Matrix**: A confusion matrix is plotted to visualize the performance of the model across the 10 classes.

## Confusion Matrix

The confusion matrix shows how well the model classifies each digit, helping to identify any misclassifications.

## Future Improvements

- Experiment with different architectures to further improve accuracy.
- Explore data augmentation techniques for more robust training.

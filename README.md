# MNIST CNN Classifier

A lightweight CNN model for MNIST digit classification that achieves >80% accuracy while maintaining less than 100,000 parameters.

## Model Architecture

The model uses a simple but effective CNN architecture:
- 2 Convolutional layers (8 -> 16 channels)
- Batch Normalization for training stability
- MaxPooling for dimension reduction
- Dropout (0.25) for regularization
- 2 Fully connected layers (784 -> 64 -> 10)

Total parameters: < 100,000

## Requirements

# MNIST CNN Classifier

[![ML Pipeline](https://github.com/basilbaby/s5_model_training/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/basilbaby/s5_model_training/actions/workflows/ml-pipeline.yml)

A lightweight CNN model for MNIST digit classification that achieves >95% accuracy while maintaining less than 25,000 parameters in a single epoch.

## Model Architecture

The model uses a simple but effective CNN architecture:
- 2 Convolutional layers (8 -> 16 channels)
- Batch Normalization for training stability
- MaxPooling for dimension reduction
- Dropout (0.1) for regularization
- 2 Fully connected layers (784 -> 64 -> 10)

Total parameters: < 25,000

## Create and activate virtual environment

```
bash
python -m venv venv
source venv/bin/activate # On Linux/Mac
or
.\venv\Scripts\activate # On Windows
```

## Install dependencies

```
pip install -r requirements.txt
```

## Training

Train the model with default settings (1 epoch):
```
PYTHONPATH=$PYTHONPATH:$(pwd) python train.py --epochs 1
```

Train for multiple epochs:
```
PYTHONPATH=$PYTHONPATH:$(pwd) python train.py --epochs 5
```

## Testing

Run the test suite:
```
PYTHONPATH=$PYTHONPATH:$(pwd) pytest tests/test_model.py -v
```


The tests verify that the model:
1. Has less than 25,000 parameters
2. Correctly handles 28x28 input images
3. Outputs 10 classes (digits 0-9)
4. Achieves >95% accuracy in one epoch
5. Handles batch processing correctly
6. Has proper gradient flow
7. Shows robustness to input noise

## Device Support
- CPU
- NVIDIA GPU (CUDA)
- Apple Silicon GPU (MPS)

The code automatically detects and uses the best available device.

## Model Saving

Trained models are saved in the `models/` directory with timestamp and device information. The saved model includes:
- Model state dict
- Optimizer state
- Device information
- Number of epochs trained
- Final test accuracy
- Parameter count

## CI/CD Pipeline

The GitHub Actions workflow:
1. Checks if model has less than 25,000 parameters
2. Trains for exactly 1 epoch
3. Verifies accuracy is above 95%
4. Uploads trained model as artifact

## Model Features

- Fast training: Achieves >95% accuracy in just one epoch
- Lightweight: Uses less than 25,000 parameters
- Efficient: Uses batch normalization and dropout for stability
- Flexible: Supports multiple compute devices
- Robust: Includes comprehensive test suite
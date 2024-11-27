# MNIST Training Pipeline

A comprehensive ML pipeline for MNIST digit classification using PyTorch, featuring hybrid CPU-GPU processing and data augmentation visualization.

## Features

- **Hybrid CPU-GPU Processing**: Efficiently utilizes both CPU and GPU for different parts of the network
- **Data Augmentation Visualization**: Visual demonstration of augmentation techniques
- **Parameter-Efficient Model**: CNN architecture with <25,000 parameters
- **Automated Pipeline**: Complete training workflow with augmentation demos
- **CI/CD Integration**: GitHub Actions workflow for automated testing and artifact collection

## Model Architecture

- 3-layer CNN with progressive channel growth (10->20->28)
- Batch Normalization and ReLU activation
- Strategic dropout (0.15)
- Hybrid computation:
  - Early layers (conv1, conv2) on GPU
  - Later layers (conv3, FC) on CPU
- Total parameters: ~23,000

## Requirements

```
bash
pip install -r requirements.txt
```

## Create and activate virtual environment

```
bash
python -m venv venv
source venv/bin/activate # On Linux/Mac
or
.\venv\Scripts\activate # On Windows
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
## Usage

### Run Complete Pipeline

```
bash
python run_pipeline.py --epochs 1 --aug-samples 3 --aug-per-sample 5
```

### Training Only

```
bash
python train.py --epochs 1
```

### Augmentation Demo Only

```
bash
python run_pipeline.py --aug-samples 3 --aug-per-sample 5
```

## Testing

Run the test suite:
```
PYTHONPATH=$PYTHONPATH:$(pwd) pytest tests/test_model.py -v
```

## Directory Structure
```
school_of_ai/
├── s5_model_training/
│ ├── model/
│ ├── utils/
│ ├── tests/
│ ├── .github/
│ ├── models/
│ ├── samples/
│ ├── requirements.txt
│ ├── train.py
│ ├── run_pipeline.py
│ └── README.md
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
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
import platform
import argparse

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    return torch.device("cpu")      # Fallback to CPU

def print_device_info():
    device = get_device()
    print("\n=== Device Information ===")
    print(f"Device Selected: {device}")
    
    if device.type == "mps":
        print("Using Apple Silicon GPU")
        print(f"PyTorch MPS Backend: {torch.backends.mps.is_available()}")
        print(f"PyTorch Version: {torch.__version__}")
    elif device.type == "cuda":
        print("Using NVIDIA GPU")
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Using CPU")
    print("========================\n")
    return device

def train(num_epochs=1):
    # Set and print device info
    device = print_device_info()
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting training for {num_epochs} epoch(s)...")
    # Train for specified number of epochs
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs} completed. Average loss: {epoch_loss:.4f}')
    
    # Save model with timestamp and device info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device_name = device.type
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'device_used': device_name,
        'epochs': num_epochs,
    }, f'models/model_{device_name}_{timestamp}.pth')
    
    print(f"\nModel saved with {device_name} configuration.")

def main():
    parser = argparse.ArgumentParser(description='Train MNIST CNN model')
    parser.add_argument('--epochs', type=int, default=1,
                      help='number of epochs to train (default: 1)')
    args = parser.parse_args()
    
    train(num_epochs=args.epochs)

if __name__ == "__main__":
    main() 
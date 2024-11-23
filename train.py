import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
import platform
import argparse
from utils.augmentation import MNISTAugmentation
from lion_pytorch import Lion

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    return torch.device("cpu")      # Fallback to CPU

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train(num_epochs=1, save_samples=False):
    # Set and print device info
    device = print_device_info()
    
    # Initialize model and print parameter count
    model = SimpleCNN().to(device)
    param_count = count_parameters(model)
    print(f"=== Model Information ===")
    print(f"Total trainable parameters: {param_count:,}")
    print(f"Parameter budget: {'OK' if param_count < 100000 else 'EXCEEDED'}")
    print("========================\n")
    
    # Initialize augmentation
    augmentation = MNISTAugmentation()
    
    # Load MNIST dataset with normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # Save augmented samples if requested
    if save_samples:
        print("\nGenerating augmented samples with predictions...")
        augmentation.save_augmented_samples(
            datasets.MNIST('data', train=True, download=True),
            model,
            device,
            num_samples=5
        )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    
    # Modified learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,  # Higher max learning rate
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # Quick warmup
        div_factor=10.0,
        final_div_factor=50.0,
        anneal_strategy='cos'
    )
    
    print(f"Starting training for {num_epochs} epoch(s)...")
    # Train for specified number of epochs
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate training accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                train_accuracy = 100 * correct / total
                print(f'Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.2f}%')
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        test_accuracy = evaluate(model, test_loader, device)
        
        print(f'Epoch {epoch+1}/{num_epochs} completed:')
        print(f'Average Loss: {epoch_loss:.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%')
        print(f'Test Accuracy: {test_accuracy:.2f}%\n')
    
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
        'final_train_accuracy': train_accuracy,
        'final_test_accuracy': test_accuracy,
        'parameters': param_count
    }, f'models/model_{device_name}_{timestamp}.pth')
    
    print(f"\nTraining completed.")
    print(f"Final training accuracy: {train_accuracy:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Model saved with {device_name} configuration.")

def main():
    parser = argparse.ArgumentParser(description='Train MNIST CNN model')
    parser.add_argument('--epochs', type=int, default=1,
                      help='number of epochs to train (default: 1)')
    parser.add_argument('--save-samples', action='store_true',
                      help='save augmented sample images')
    args = parser.parse_args()
    
    train(num_epochs=args.epochs, save_samples=args.save_samples)

if __name__ == "__main__":
    main() 
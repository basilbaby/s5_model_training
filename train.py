import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
import platform
import argparse

def get_devices():
    devices = {
        'cpu': torch.device('cpu'),
        'gpu': None
    }
    # Check for GPU availability
    if torch.backends.mps.is_available():
        devices['gpu'] = torch.device("mps")
    elif torch.cuda.is_available():
        devices['gpu'] = torch.device("cuda")
    return devices

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            if device:
                data, target = data.to(device), target.to(device)
            outputs = model(data)
            if device:
                outputs = outputs.cpu()
                target = target.cpu()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train(num_epochs=1):
    # Get both devices
    devices = get_devices()
    print("\n=== Device Information ===")
    print(f"CPU: {devices['cpu']}")
    print(f"GPU: {devices['gpu']}")
    print("========================\n")
    
    # Initialize model and move to appropriate devices
    model = SimpleCNN()
    model.to_devices(devices)
    
    # Print model parameter count
    param_count = count_parameters(model)
    print("\n=== Model Information ===")
    print(f"Total trainable parameters: {param_count:,}")
    print(f"Parameter budget: {'OK' if param_count < 25000 else 'EXCEEDED'}")
    print("========================\n")
    
    # Fixed transform order - ToTensor() must come before RandomErasing
    transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=7,
            translate=(0.07, 0.07),
            scale=(0.93, 1.07)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        amsgrad=True
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.25,
        div_factor=10.0,
        final_div_factor=100.0,
        anneal_strategy='cos'
    )
    
    print(f"Starting training for {num_epochs} epoch(s)...")
    # Train for specified number of epochs
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move input to GPU if available
        if devices['gpu']:
            data = data.to(devices['gpu'])
        
        # Forward pass
        output = model(data)  # No need to pass devices anymore
        
        # Loss calculation on CPU
        target = target.cpu()
        output = output.cpu()
        loss = criterion(output, target)
        
        # Backward and optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate metrics
        with torch.no_grad():
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        running_loss += loss.item()
        if batch_idx % 20 == 0:
            train_accuracy = 100 * correct / total
            print(f'Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Training Accuracy: {train_accuracy:.2f}%, '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    test_accuracy = evaluate(model, test_loader, devices['gpu'])
    
    print(f'Epoch {num_epochs} completed:')
    print(f'Average Loss: {epoch_loss:.4f}')
    print(f'Training Accuracy: {train_accuracy:.2f}%')
    print(f'Test Accuracy: {test_accuracy:.2f}%\n')
    
    # Save model with timestamp and device info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device_name = devices['gpu'].type if devices['gpu'] else 'cpu'
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'device_used': device_name,
        'epochs': num_epochs,
        'final_test_accuracy': test_accuracy,
        'parameters': count_parameters(model)
    }, f'models/model_{device_name}_{timestamp}.pth')
    
    print(f"\nTraining completed. Final test accuracy: {test_accuracy:.2f}%")
    print(f"Model saved with {device_name} configuration.")

def main():
    parser = argparse.ArgumentParser(description='Train MNIST CNN model')
    parser.add_argument('--epochs', type=int, default=1,
                      help='number of epochs to train (default: 1)')
    args = parser.parse_args()
    
    train(num_epochs=args.epochs)

if __name__ == "__main__":
    main() 
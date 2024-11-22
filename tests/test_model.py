import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from model.network import SimpleCNN
from torchvision import datasets, transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = SimpleCNN()
    param_count = count_parameters(model)
    print(f"\nTotal trainable parameters: {param_count}")
    assert param_count < 100000, f"Model has {param_count} parameters, which exceeds the limit of 100,000"

def test_input_output_dimensions():
    model = SimpleCNN()
    # Create a sample input of size [1, 1, 28, 28]
    sample_input = torch.randn(1, 1, 28, 28)
    
    # Test forward pass
    with torch.no_grad():
        output = model(sample_input)
    
    # Check output dimensions
    assert output.shape[1] == 10, f"Model output should have 10 classes, but got {output.shape[1]}"
    print(f"\nOutput shape: {output.shape}")

def test_model_accuracy():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Load training dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Train the model for 2 epochs
    print("\nTraining model for accuracy test...")
    model.train()
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Training Epoch {epoch+1}/2 - Batch {batch_idx}/{len(train_loader)}')
    
    # Test the model
    print("\nEvaluating model...")
    model.eval()
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
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
    print(f"\nAccuracy on test set: {accuracy:.2f}%")
    assert accuracy > 80, f"Model accuracy is {accuracy:.2f}%, which is below the required 80%"

if __name__ == "__main__":
    pytest.main([__file__]) 
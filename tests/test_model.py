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
    # Create a sample input with batch size 4
    sample_input = torch.randn(4, 1, 28, 28)
    
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

# New test 1: Test model behavior with batch processing
def test_batch_processing():
    model = SimpleCNN()
    batch_size = 32
    sample_batch = torch.randn(batch_size, 1, 28, 28)
    
    with torch.no_grad():
        output = model(sample_batch)
    
    assert output.shape[0] == batch_size, f"Batch size mismatch: expected {batch_size}, got {output.shape[0]}"
    assert output.shape[1] == 10, "Output classes mismatch"
    print(f"\nBatch processing test passed with batch size {batch_size}")

# New test 2: Test model gradient flow
def test_gradient_flow():
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Sample data with batch size 4
    inputs = torch.randn(4, 1, 28, 28)
    targets = torch.tensor([5, 3, 2, 1])  # Random target classes
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Check if gradients are computed
    has_gradients = all(param.grad is not None for param in model.parameters() if param.requires_grad)
    assert has_gradients, "Some parameters do not have gradients"
    
    # Check if gradients are non-zero
    has_nonzero_gradients = any(param.grad.abs().sum() > 0 for param in model.parameters() if param.requires_grad)
    assert has_nonzero_gradients, "All gradients are zero"
    print("\nGradient flow test passed")

# New test 3: Test model robustness to noise
def test_noise_robustness():
    model = SimpleCNN()
    model.eval()
    
    # Generate clean sample
    clean_input = torch.randn(1, 1, 28, 28)
    
    # Generate noisy sample
    noise = torch.randn_like(clean_input) * 0.1
    noisy_input = clean_input + noise
    
    with torch.no_grad():
        clean_output = model(clean_input)
        noisy_output = model(noisy_input)
    
    # Check if predictions are consistent
    _, clean_pred = torch.max(clean_output, 1)
    _, noisy_pred = torch.max(noisy_output, 1)
    
    # Calculate output difference
    output_diff = (clean_output - noisy_output).abs().mean().item()
    assert output_diff < 0.5, f"Model is too sensitive to noise: mean difference = {output_diff}"
    print(f"\nNoise robustness test passed with mean difference: {output_diff:.4f}")

if __name__ == "__main__":
    pytest.main([__file__]) 
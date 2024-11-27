import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First Convolution Block - slightly reduced
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)  # 12 -> 10
        self.bn1 = nn.BatchNorm2d(10)
        
        # Second Convolution Block - slightly reduced
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)  # 24 -> 20
        self.bn2 = nn.BatchNorm2d(20)
        
        # Third Convolution Block - slightly reduced
        self.conv3 = nn.Conv2d(20, 28, kernel_size=3, stride=1, padding=1)  # 32 -> 28
        self.bn3 = nn.BatchNorm2d(28)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully Connected layers - slightly reduced
        self.fc1 = nn.Linear(28 * 3 * 3, 56)  # 64 -> 56
        self.fc2 = nn.Linear(56, 10)
        
        # Dropout
        self.dropout = nn.Dropout(0.15)  # Keep same dropout rate

    def to_devices(self, devices):
        if devices['gpu']:
            # Move early layers to GPU
            self.conv1 = self.conv1.to(devices['gpu'])
            self.bn1 = self.bn1.to(devices['gpu'])
            self.conv2 = self.conv2.to(devices['gpu'])
            self.bn2 = self.bn2.to(devices['gpu'])
        
        # Keep later layers on CPU
        self.conv3 = self.conv3.to(devices['cpu'])
        self.bn3 = self.bn3.to(devices['cpu'])
        self.fc1 = self.fc1.to(devices['cpu'])
        self.fc2 = self.fc2.to(devices['cpu'])
        return self

    def forward(self, x):
        # First block on GPU if available
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second block on GPU if available
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Move to CPU for remaining operations
        x = x.cpu()
        
        # Third block on CPU
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten and FC layers on CPU
        x = x.view(-1, 28 * 3 * 3)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
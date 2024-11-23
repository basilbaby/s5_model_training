import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

class MNISTAugmentation:
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # For visualization (without normalization)
        self.viz_transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor()
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def save_augmented_samples(self, dataset, model, device, num_samples=5):
        """Save original and augmented versions of sample images with model predictions"""
        if not os.path.exists('samples'):
            os.makedirs('samples')
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.eval()
        
        plt.figure(figsize=(15, 3*num_samples))
        
        for i in range(num_samples):
            image, label = dataset[i]
            
            # Get multiple augmented versions
            augmented_images = [self.viz_transform(image) for _ in range(3)]
            all_images = [transforms.ToTensor()(image)] + augmented_images
            
            # Get model predictions
            with torch.no_grad():
                # Normalize images for model input
                norm_images = [transforms.Normalize((0.1307,), (0.3081,))(img.clone()) for img in all_images]
                predictions = []
                for img in norm_images:
                    output = model(img.unsqueeze(0).to(device))
                    pred = output.argmax(dim=1).item()
                    confidence = torch.nn.functional.softmax(output, dim=1).max().item()
                    predictions.append((pred, confidence))
            
            # Create subplot for this sample
            plt.subplot(num_samples, 1, i+1)
            
            # Create grid of images
            img_grid = make_grid(all_images, nrow=4, padding=5, normalize=True)
            plt.imshow(img_grid.permute(1, 2, 0))
            
            # Add titles with predictions
            title = f'True Label: {label} | '
            title += ' | '.join([f'Pred: {pred} ({conf:.2f})' for pred, conf in predictions])
            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'samples/augmentation_with_predictions_{timestamp}.png')
        plt.close()
            
        print(f"Saved {num_samples} augmented samples with predictions in 'samples' directory")
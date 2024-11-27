import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from datetime import datetime

class AugmentationDemo:
    def __init__(self):
        # Define base transformations
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Define augmentation transformations
        self.aug_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=7,
                translate=(0.07, 0.07),
                scale=(0.93, 1.07)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def denormalize(self, tensor):
        """Denormalize the tensor for visualization"""
        tensor = tensor.clone()
        tensor = tensor * 0.3081 + 0.1307
        tensor = torch.clamp(tensor, 0, 1)
        return tensor

    def visualize_augmentations(self, image, num_augmentations=5, save_path='samples'):
        """Generate and visualize multiple augmentations of an image"""
        # Create samples directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Convert to PIL Image if tensor
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
            
        # Create a figure
        fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(2*(num_augmentations + 1), 2))
        
        # Plot original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Generate and plot augmentations
        for i in range(num_augmentations):
            # Apply augmentation
            aug_img = self.aug_transform(image)
            # Denormalize for visualization
            aug_img = self.denormalize(aug_img)
            # Convert to numpy and squeeze
            aug_img = aug_img.squeeze().numpy()
            
            # Plot
            axes[i+1].imshow(aug_img, cmap='gray')
            axes[i+1].set_title(f'Aug {i+1}')
            axes[i+1].axis('off')
        
        # Save the figure
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = os.path.join(save_path, f'augmentations_{timestamp}.png')
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Augmentations saved to {save_file}")
        return save_file

def demo_augmentations(dataset, num_samples=3, num_augmentations=5):
    """Demo augmentations on multiple samples from the dataset"""
    aug_demo = AugmentationDemo()
    
    for i in range(num_samples):
        # Get a random image from dataset
        image, _ = dataset[torch.randint(len(dataset), (1,)).item()]
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Visualize augmentations
        aug_demo.visualize_augmentations(
            image, 
            num_augmentations=num_augmentations,
            save_path='samples'
        ) 
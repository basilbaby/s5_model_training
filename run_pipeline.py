import argparse
from train import train
from utils.augmentation import demo_augmentations
from torchvision import datasets, transforms
import os

def run_pipeline(epochs=1, num_aug_samples=3, num_augmentations=5):
    print("\n=== Starting ML Pipeline ===")
    
    # Step 1: Run Augmentation Demo
    print("\n1. Running Augmentation Visualization...")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    aug_files = demo_augmentations(dataset, 
                                 num_samples=num_aug_samples,
                                 num_augmentations=num_augmentations)
    print("Augmentation visualization completed.")
    
    # Step 2: Train Model
    print("\n2. Starting Model Training...")
    train(num_epochs=epochs)
    
    # Step 3: Print Pipeline Summary
    print("\n=== Pipeline Summary ===")
    print(f"- Augmentation samples generated: {num_aug_samples}")
    print(f"- Augmentations per sample: {num_augmentations}")
    print(f"- Training epochs completed: {epochs}")
    print(f"- Augmentation samples saved in: {os.path.abspath('samples')}")
    print(f"- Model checkpoints saved in: {os.path.abspath('models')}")
    print("======================\n")

def main():
    parser = argparse.ArgumentParser(description='Run ML Pipeline with Augmentation and Training')
    parser.add_argument('--epochs', type=int, default=1,
                      help='number of epochs to train (default: 1)')
    parser.add_argument('--aug-samples', type=int, default=3,
                      help='number of samples for augmentation demo (default: 3)')
    parser.add_argument('--aug-per-sample', type=int, default=5,
                      help='number of augmentations per sample (default: 5)')
    
    args = parser.parse_args()
    
    run_pipeline(
        epochs=args.epochs,
        num_aug_samples=args.aug_samples,
        num_augmentations=args.aug_per_sample
    )

if __name__ == "__main__":
    main() 
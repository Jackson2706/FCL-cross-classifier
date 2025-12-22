import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import shutil

# Download CIFAR-10 dataset to current directory
transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform)

# Create a dictionary to store images by class
cifar10_classes = {i: [] for i in range(10)}

# Process train images
for img, label in trainset:
    cifar10_classes[label].append(img.numpy())

# Process test images
for img, label in testset:
    cifar10_classes[label].append(img.numpy())

# Create output folder
output_folder = "cifar10-classes"
os.makedirs(output_folder, exist_ok=True)

# Save each class as a .npy file
for class_id, images in cifar10_classes.items():
    images = np.array(images, dtype=np.float32)  # Convert list to numpy array
    save_path = os.path.join(output_folder, f"{class_id}.npy")
    np.save(save_path, images)
    print(f"Saved {save_path} with {len(images)} images.")

print("CIFAR-10 saved in 'cifar10-classes' folder.")

# Delete .gz files and data files after extraction (if any)
gz_files = [f for f in os.listdir('.') if f.endswith('.gz')]
for gz_file in gz_files:
    gz_path = os.path.join('.', gz_file)
    os.remove(gz_path)
    print(f"Deleted {gz_path}")

# Delete extracted CIFAR-10 data directory
extracted_folder = 'cifar-10-batches-py'
if os.path.exists(extracted_folder):
    shutil.rmtree(extracted_folder)
    print(f"Deleted {extracted_folder}")

print("Cleanup done.")

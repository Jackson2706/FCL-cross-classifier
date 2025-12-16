import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import shutil

# Download CIFAR-100 dataset
transform = transforms.ToTensor()
trainset = torchvision.datasets.CIFAR100(root='dataset', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='dataset', train=False, download=True, transform=transform)

# Create a dictionary to store images by class
cifar100_classes = {i: [] for i in range(100)}

# Process train images
for img, label in trainset:
    cifar100_classes[label].append(img.numpy())

# Process test images
for img, label in testset:
    cifar100_classes[label].append(img.numpy())

# Create output folder
output_folder = "dataset/cifar100-classes"
os.makedirs(output_folder, exist_ok=True)

# Save each class as a .npy file
for class_id, images in cifar100_classes.items():
    images = np.array(images, dtype=np.float32)  # Convert list to numpy array
    save_path = os.path.join(output_folder, f"{class_id}.npy")
    np.save(save_path, images)
    print(f"Saved {save_path} with {len(images)} images.")

print("CIFAR-100 saved in 'cifar100-classes' folder.")

# Delete .gz files and data files after extraction
gz_files = [f for f in os.listdir('dataset') if f.endswith('.gz')]
for gz_file in gz_files:
    gz_path = os.path.join('dataset', gz_file)
    os.remove(gz_path)
    print(f"Deleted {gz_path}")

# Delete extracted data files
extracted_files = [
    'dataset/cifar-100-python'
]

for file_path in extracted_files:
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
        print(f"Deleted {file_path}")

print("Deleted .gz files and extracted data files.")

import gc
import os
import pickle
import time
from typing import Any
from torch.utils.data import Subset, Dataset
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from wilds import get_dataset

img_size = 32
cifar100_train_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.RandomCrop((img_size, img_size), padding=4),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.ColorJitter(brightness=0.24705882352941178),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
cifar100_test_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

# Mean và std của ImageNet1k gốc
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

imagenet_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.ColorJitter(
    #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)])

imagenet_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),  
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)])


# client train data 1 task
def read_client_data_FCL_imagenet1k(index, task = 0, classes_per_task = 2, count_labels=False, train=True):
    
    datadir = '/media/jackson/Data/FedGCC/dataset/imagenet1k-classes/'
    class_order = np.load('/media/jackson/Data/FedGCC/dataset/class_order/class_order_imagenet1k.npy', allow_pickle=True)

    class_order = class_order[index]

    if train:
        x, y = load_data(datadir, class_order[task*classes_per_task:(task+1)*classes_per_task], train=True)
    else:
        x, y = load_data(datadir, class_order[task*classes_per_task:(task+1)*classes_per_task], train=False)
    x = x.type(torch.FloatTensor)
    y = torch.Tensor(y.type(torch.long))
    data = Transform_dataset(x, y, imagenet_train_transform if train else imagenet_test_transform)

    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y, return_counts=True)
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts

        return data, label_info
    
    return data


def read_client_data_FCL_cifar100(
    index, 
    task=0, 
    classes_per_task=2, 
    count_labels=False, 
    train=True,
    datadir='/media/jackson/Data/FedGCC/dataset/cifar100-classes/',
    class_order_path='/media/jackson/Data/FedGCC/dataset/class_order/class_order_cifar100.npy',
    train_images_per_class=500,
    test_images_per_class=100
):
    """
    Reads client data for Federated Continual Learning.
    Flexible for any number of clients, tasks, and classes per task.
    """
    # 1. Load class orders
    all_class_orders = np.load(class_order_path, allow_pickle=True)
    
    # 2. FLEXIBILITY: Number of Clients
    # Using modulo operator (%) ensures that if you have more clients than 
    # pre-defined class orders, it wraps around without throwing an IndexError.
    client_class_order = all_class_orders[index % len(all_class_orders)]

    # 3. FLEXIBILITY: Number of Tasks & Classes per Task
    # Calculate slice indices and ensure we don't go out of bounds
    start_idx = task * classes_per_task
    end_idx = (task + 1) * classes_per_task
    
    if start_idx >= len(client_class_order):
        raise ValueError(f"Task index {task} is out of bounds. The client only has {len(client_class_order)} classes assigned.")
    
    target_classes = client_class_order[start_idx:end_idx]

    # 4. Load the actual data using parameterized variables
    x, y = load_data(
        datadir, 
        target_classes, 
        train_images_per_class=train_images_per_class, 
        test_images_per_class=test_images_per_class, 
        train=train
    )
    
    x = x.type(torch.FloatTensor)
    y = torch.Tensor(y.type(torch.long))
    data = Transform_dataset(x, y)

    # 5. Optional: Count labels
    if count_labels:
        label_info = {}
        unique_y, counts = torch.unique(y, return_counts=True)
        label_info['labels'] = unique_y.detach().numpy()
        label_info['counts'] = counts.detach().numpy()

        return data, label_info
    
    return data

def read_client_data_FCL_cifar10(index, task = 0, classes_per_task = 2, count_labels=False, train=True):
    
    datadir = '/media/jackson/Data/FedGCC/dataset/cifar10-classes/'
    class_order = np.load('/media/jackson/Data/FedGCC/dataset/class_order/class_order_cifar10.npy', allow_pickle=True)
    class_order = class_order[index]

    if train:
        x, y = load_data(datadir, class_order[task*classes_per_task:(task+1)*classes_per_task], train_images_per_class=5000, test_images_per_class=1000, train=True)
    else:
        x, y = load_data(datadir, class_order[task*classes_per_task:(task+1)*classes_per_task], train_images_per_class=5000, test_images_per_class=1000, train=False)
    # x = x.type(torch.FloatTensor)
    # print(x.shape)
    y = torch.Tensor(y.type(torch.long))
    data = Transform_dataset(x, y)

    if count_labels:
        label_info = {}
        unique_y, counts=torch.unique(y, return_counts=True)
        unique_y=unique_y.detach().numpy()
        counts=counts.detach().numpy()
        label_info['labels']=unique_y
        label_info['counts']=counts

        return data, label_info
    
    return data

class Transform_dataset(data.Dataset):
    def __init__(self, X, Y, transform=None) -> None:
        super().__init__()
        self.X = X
        self.Y = Y
        self.transform = transform
    
    def __getitem__(self, index: Any) -> Any:
        x = self.X[index]
        y = self.Y[index]
        if self.transform:
            x = self.transform(x)
        return x,y

    def __len__(self) -> int:
        return len(self.X)
    
def load_data(datadir, classes=[], train_images_per_class = 600, test_images_per_class = 100, train=True):
    x, y = [], []

    for _class in classes:
        data_file = datadir + str(_class) + '.npy'
        data = np.load(data_file)
        
        if train:
            x.append(data[:train_images_per_class])
            y.append(np.array([_class] * train_images_per_class))
        else:
            x.append(data[train_images_per_class:])
            y.append(np.array([_class] * test_images_per_class))
    x = torch.tensor(np.concatenate(x))
    y = torch.from_numpy(np.concatenate(y))
    return x, y


def load_full_test_data(datadir, dataset="IMAGENET1k", train_images_per_class=5000, test_images_per_class=1000, concat_every=5):
    if dataset == "CIFAR100":
        classes = list(range(100))
    elif dataset == "CIFAR10":
        classes = list(range(10))
    elif dataset == "IMAGENET1k":
        classes = list(range(1000))
    else:
        raise NotImplementedError("Not supported dataset")

    x_test_all = None
    y_test_all = None

    x_test_list, y_test_list = [], []

    for i, _class in enumerate(classes):
        print(f"Loading data full for class {_class}...")
        data_file = datadir + str(_class) + '.npy'
        data = np.load(data_file)  

        new_x_test = data[train_images_per_class:]  
        new_y_test = np.full((test_images_per_class,), _class, dtype=np.int64)

        x_test_list.append(new_x_test)
        y_test_list.append(new_y_test)

        # Mỗi concat_every class thì concat để giảm bộ nhớ
        if (i + 1) % concat_every == 0 or (i + 1) == len(classes):

            if x_test_all is None:
                x_test_all = np.concatenate(x_test_list, axis=0)
                y_test_all = np.concatenate(y_test_list, axis=0)
            else:
                x_test_all = np.concatenate([x_test_all] + x_test_list, axis=0)
                y_test_all = np.concatenate([y_test_all] + y_test_list, axis=0)

            # Giải phóng bộ nhớ
            x_test_list.clear()
            y_test_list.clear()
            gc.collect()

    # Convert sang tensor
    x_test_tensor = torch.tensor(x_test_all, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_all, dtype=torch.long)

    print(f"Loaded {len(x_test_tensor)} test images and {len(y_test_tensor)} test labels.")
    return x_test_tensor, y_test_tensor


def load_full_data(datadir, dataset="IMAGENET1k", train_images_per_class=600, test_images_per_class=100, concat_every=200):
    if dataset == "CIFAR100":
        classes = list(range(100))
    elif dataset == "IMAGENET1k":
        classes = list(range(1000))
    else:
        raise NotImplementedError("Not supported dataset")

    x_train_all = None
    y_train_all = None
    x_test_all = None
    y_test_all = None

    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    for i, _class in enumerate(classes):
        print(f"Loading data full for class {_class}...")
        data_file = datadir + str(_class) + '.npy'
        data = np.load(data_file)  

        new_x_train = data[:train_images_per_class]
        new_x_test = data[train_images_per_class:]  

        new_y_train = np.full((train_images_per_class,), _class, dtype=np.int64)
        new_y_test = np.full((test_images_per_class,), _class, dtype=np.int64)

        x_train_list.append(new_x_train)
        y_train_list.append(new_y_train)
        x_test_list.append(new_x_test)
        y_test_list.append(new_y_test)

        # Mỗi concat_every class thì concat để giảm bộ nhớ
        if (i + 1) % concat_every == 0 or (i + 1) == len(classes):
            print(f"Concatenating batch of {len(x_train_list)} classes...")

            if x_train_all is None:
                x_train_all = np.concatenate(x_train_list, axis=0)
                y_train_all = np.concatenate(y_train_list, axis=0)
                x_test_all = np.concatenate(x_test_list, axis=0)
                y_test_all = np.concatenate(y_test_list, axis=0)
            else:
                x_train_all = np.concatenate([x_train_all] + x_train_list, axis=0)
                y_train_all = np.concatenate([y_train_all] + y_train_list, axis=0)
                x_test_all = np.concatenate([x_test_all] + x_test_list, axis=0)
                y_test_all = np.concatenate([y_test_all] + y_test_list, axis=0)

            # Giải phóng bộ nhớ
            x_train_list.clear()
            y_train_list.clear()
            x_test_list.clear()
            y_test_list.clear()
            gc.collect()

    # Convert sang tensor
    x_train_tensor = torch.tensor(x_train_all, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_all, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test_all, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_all, dtype=torch.long)

    print(f"Loaded {len(x_train_tensor)} training and {len(x_test_tensor)} test images.")
    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor

def get_unique_tasks(task_list):
    unique_tasks = {tuple(sorted(task)) for task in task_list}
    return [list(task) for task in unique_tasks]

# datadir = '/root/projects/FCL/dataset/imagenet1k-classes/'
# load_test_data(datadir, dataset='IMAGENET1k', train_images_per_class=600, test_images_per_class=50)

# test_data_all_task = []
# for task in range(500):    
#     _, test_data, _ = read_client_data_FCL_imagenet1k(1, task=task, classes_per_task=2, count_labels=True)
    
#     test_data_all_task.append(test_data)
#     print(task)

# ==========================================
# 1. WILDS Transforms (No ToPILImage!)
# ==========================================
img_size_wilds = 32
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

wilds_train_transform = transforms.Compose([
    transforms.Resize((img_size_wilds, img_size_wilds)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

wilds_test_transform = transforms.Compose([
    transforms.Resize((img_size_wilds, img_size_wilds)),  
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# ==========================================
# 2. WILDS Dataset Wrapper
# ==========================================
class WILDS_FCL_Dataset(Dataset):
    """Strips metadata to return standard (x, y) for FCL loops."""
    def __init__(self, wilds_subset, transform=None):
        self.wilds_subset = wilds_subset
        self.transform = transform

    def __len__(self):
        return len(self.wilds_subset)

    def __getitem__(self, index):
        x, y, metadata = self.wilds_subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

# ==========================================
# 3. Main Data Loader (Handles 5 Tasks/Domains)
# ==========================================
def read_client_data_FCL_wilds(index, num_clients, task=0, dataset_name="camelyon17", count_labels=False, train=True):
    # Make sure this points to your downloaded dataset folder
    datadir = '/media/jackson/Data/FedGCC/dataset/wilds/' 
    
    # 1. Initialize dataset
    dataset = get_dataset(dataset=dataset_name, download=False, root_dir=datadir)
    
    # 2. Get the correct split
    try:
        split_name = 'train' if train else 'test'
        subset = dataset.get_subset(split_name)
    except ValueError:
        split_name = 'val' if not train else 'train'
        subset = dataset.get_subset(split_name)

    # 3. Extract domains
    domain_idx = 0 
    domains = subset.metadata_array[:, domain_idx]
    unique_domains = torch.unique(domains) # Camelyon17 has exactly 5 unique domains
    
    # 4. Task Logic (0 to 4): Shift the domain based on the current task
    target_domain_idx = (index + task) % len(unique_domains)
    target_domain = unique_domains[target_domain_idx]
    
    # 5. Filter and Partition
    domain_indices = (domains == target_domain).nonzero(as_tuple=True)[0]
    
    # Seeded shuffle ensures clients don't overlap data
    generator = torch.Generator().manual_seed(42 + target_domain.item())
    shuffled_indices = domain_indices[torch.randperm(len(domain_indices), generator=generator)]
    
    chunk_size = len(shuffled_indices) // num_clients
    start_idx = index * chunk_size
    end_idx = len(shuffled_indices) if index == num_clients - 1 else start_idx + chunk_size
        
    task_indices = shuffled_indices[start_idx:end_idx]
    task_subset = Subset(subset, task_indices.tolist())
    
    # 6. Apply Transforms
    transform_to_use = wilds_train_transform if train else wilds_test_transform
    client_data = WILDS_FCL_Dataset(task_subset, transform=transform_to_use)

    # 7. Count Labels (Fixed for WILDS subsets)
    if count_labels:
        label_info = {}
        y_vals = subset.y_array[task_indices] 
        
        if not isinstance(y_vals, torch.Tensor):
            y_vals = torch.tensor(y_vals)
            
        unique_y, counts = torch.unique(y_vals, return_counts=True)
        
        label_info['labels'] = unique_y.detach().cpu().numpy()
        label_info['counts'] = counts.detach().cpu().numpy()
        
        return client_data, label_info
        
    return client_data


import os
def download_wilds_dataset(dataset_name="camelyon17", datadir='/media/jackson/Data/FedGCC/dataset/wilds/'):
    """
    Downloads the specified WILDS dataset to the target directory.
    """
    print(f"Preparing to download: {dataset_name}")
    print(f"Target directory: {datadir}")
    
    # Ensure the directory exists
    os.makedirs(datadir, exist_ok=True)
    
    # The download=True flag handles downloading and extracting
    dataset = get_dataset(dataset=dataset_name, download=True, root_dir=datadir)
    
    print(f"\nSuccess! {dataset_name} is ready.")
    print(f"Total number of samples in the dataset: {len(dataset)}")

if __name__ == "__main__":
    # You can change "camelyon17" to "iwildcam" or others if needed
    download_wilds_dataset("camelyon17")
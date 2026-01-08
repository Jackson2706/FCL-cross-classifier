import copy
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
from flcore.clients.clientbase import Client
from flcore.utils_core.target_utils import Generator
from torch.utils.data import DataLoader, TensorDataset


class clientOurs(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        # --- 1. Cấu hình Generator ---
        # Tự động xác định cấu hình dựa trên tên dataset
        if 'cifar' in args.dataset.lower():
            self.img_size = 32
            self.nz = 256
        elif 'imagenet' in args.dataset.lower():
            self.img_size = 64 if '100' in args.dataset else 224
            self.nz = 256
        else:
            self.img_size = 64
            self.nz = 100
        
        self.nc = 3 # RGB channels

        # Khởi tạo kiến trúc Generator (Trọng số sẽ được Server ghi đè)
        self.generator = None

        # Tham số Replay
        self.replay_weight = args.replay_weight if hasattr(args, 'replay_weight') else 1.0

    def train(self, task):
        
        trainloader = self.load_train_data(task=task)
        # Check samples per class in real data
        self.class_sample_count = self._count_samples_per_class(trainloader.dataset)
        self.real_classes = set(self.class_sample_count.keys())
        print(f"[Client {self.id}] Real classes in task {task}: {self.real_classes}")
        if self.generator:
            print(f"[Client {self.id}] Generator available for classes: {self.get_generator_classes()}")
            trainloader = self.create_augmented_dataset(trainloader.dataset)

        start_time = time.time()
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        local_loss = []
        for epoch in range(self.local_epochs):
            for data, target in trainloader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                local_loss.append(loss.item())
                self.optimizer.step()
        train_time = time.time() - start_time

        # if self.learning_rate_scheduler:
        #     self.learning_rate_scheduler.step()
        
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"[Client {self.id}] Task {task} | Loss: {sum(local_loss)/len(local_loss):.4f} | LR: {current_lr:.6f}")
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += train_time

    def freeze_generator(self):
        """Đảm bảo Generator không bao giờ được train ở Client"""
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def set_generator_parameters(self, global_generator):
        """
        Nhận Generator mới nhất từ Server
        """
        if self.generator:
            self.generator.load_state_dict(global_generator.state_dict())
        else:
            self.generator = global_generator
        self.freeze_generator()
        self.generator.to(self.device)
        
    def _count_samples_per_class(self, dataset):
        """Count number of samples per class in training data"""
        class_counts = {}
        for _, label in dataset:
            label = int(label)
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts
        
    def get_generator_classes(self):
        """Get classes that generator can generate"""
        # Assuming generator has num_classes attribute or infer from output layer
        if hasattr(self.generator, 'num_classes'):
            return set(range(self.generator.num_classes))
        return set(range(10))  # Default fallback
        
    def create_augmented_dataset(self, train_data):
        """Create dataset with real data + generated data for missing classes"""
        generator_classes = self.get_generator_classes()
        missing_classes = generator_classes - self.real_classes
        
        # Store for use in training
        self.real_classes = self.real_classes
        self.missing_classes = missing_classes
        self.all_classes = self.real_classes | missing_classes

        # Generate synthetic samples for missing classes
        synthetic_data = []
        synthetic_labels = []

        if missing_classes:
            for class_id in missing_classes:
                num_samples = max(self.class_sample_count.values()) if self.class_sample_count else 100
                z = torch.randn(num_samples, self.nz)
                class_labels = torch.full((num_samples,), class_id, dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    synthetic_samples = self.generator(z, class_labels)
                
                synthetic_data.append(synthetic_samples.cpu())
                synthetic_labels.extend([class_id] * num_samples)

        # Combine real and synthetic data
        combined_data = list(train_data)
        if synthetic_data:
            synthetic_data_tensor = torch.cat(synthetic_data, dim=0)
            for img, label in zip(synthetic_data_tensor, synthetic_labels):
                combined_data.append((img, label))

        # Create DataLoader
        return DataLoader(combined_data, batch_size=self.batch_size, shuffle=True)
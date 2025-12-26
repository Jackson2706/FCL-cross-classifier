import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from flcore.utils_core.target_utils import *

# Assuming Generator is defined in flcore.trainmodel.models

class clientOursV2(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        
        # --- Config Generator ---
        if 'cifar100' in args.dataset.lower():
            self.img_size = 32; self.nz = 256
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)
        elif 'imagenet' in args.dataset.lower():
            self.img_size = 64 if '100' in args.dataset else 224; self.nz = 256
        else:
            self.img_size = 32; self.nz = 100
            
        self.nc = 3
        
        # --- Hyperparameters ---
        self.replay_weight = getattr(args, 'replay_weight', 1.0)
        self.T = getattr(args, 'T', 2.0) 

        # --- Initialize Generator ---
        self.generator = None

    def freeze_generator(self):
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


    def get_feature_embeddings(self, model=None, task_id=None, num_samples=None):
        """Extracts feature embeddings for t-SNE visualization."""
        if task_id is None: task_id = self.current_task
        target_model = model if model is not None else self.model
        dataloader = self.load_test_data(task=task_id, batch_size=self.batch_size)
        
        target_model.eval()
        target_model.to(self.device)
        
        features_list, labels_list = [], []
        count = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                if isinstance(x, list): x = x[0]
                x, y = x.to(self.device), y.to(self.device)
                
                # Extract Feature
                if hasattr(target_model, 'base'):
                    feats = target_model.base(x)
                else:
                    feats = x
                    modules = list(target_model.children())[:-1]
                    feats = nn.Sequential(*modules)(x).view(x.size(0), -1)
                
                features_list.append(feats.detach().cpu())
                labels_list.append(y.detach().cpu())
                
                count += len(y)
                if num_samples and count >= num_samples: break
                
        if len(features_list) > 0:
            return torch.cat(features_list), torch.cat(labels_list)
        return torch.tensor([]), torch.tensor([])

    def train(self, task=None):
        trainloader = self.load_train_data(task=task)
        self.class_sample_count = self._count_samples_per_class(trainloader.dataset)
        self.real_classes = set(self.class_sample_count.keys())
        print(f"[Client {self.id}] Real classes in task {task}: {self.real_classes}")
        if self.generator:
            self.generator.eval()
            print(f"[Client {self.id}] Generator available for classes: {self.get_generator_classes()}")
            trainloader = self.create_augmented_dataset(trainloader.dataset)
        
        # Ensure model is trainable
        for param in self.model.parameters(): param.requires_grad = True
        self.model.train()
        
        start_time = time.time()
        local_loss = []
        
        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                self.optimizer.zero_grad()
                if isinstance(x, list): x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

               
                local_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            if self.learning_rate_scheduler:
                self.learning_rate_scheduler.step()
                
        # Logging
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"[Client {self.id}] Task {task} | Loss: {sum(local_loss)/len(local_loss):.4f} | LR: {current_lr:.6f}")
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    
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

        # --- [FIX START] Analyze Real Data Structure ---
        # We check the first sample to see if labels are Tensors or Ints
        # and if there are extra return values (like sample indices).
        if len(train_data) > 0:
            sample_real = train_data[0] # Get first sample
            # Check if label (index 1) is a Tensor
            is_label_tensor = isinstance(sample_real[1], torch.Tensor)
            # Check tuple length (e.g., if data returns (img, label, index))
            data_tuple_len = len(sample_real)
        else:
            # Default fallback
            is_label_tensor = True
            data_tuple_len = 2
        # --- [FIX END] ---

        # Generate synthetic samples for missing classes
        synthetic_data = []
        synthetic_labels = []

        if missing_classes:
            for class_id in missing_classes:
                num_samples = max(self.class_sample_count.values()) if self.class_sample_count else 100
                z = torch.randn(num_samples, self.nz).to(self.device)
                class_labels = torch.full((num_samples,), class_id, dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    synthetic_samples = self.generator(z, class_labels)
                
                synthetic_data.append(synthetic_samples.cpu())
                synthetic_labels.extend([class_id] * num_samples)

        # Combine real and synthetic data
        combined_data = list(train_data)
        
        if synthetic_data:
            synthetic_data_tensor = torch.cat(synthetic_data, dim=0)
            
            # --- [FIX START] Append Synthetic Data with Correct Format ---
            for i, (img, label_int) in enumerate(zip(synthetic_data_tensor, synthetic_labels)):
                
                # 1. Match Label Type
                if is_label_tensor:
                    label = torch.tensor(label_int, dtype=torch.long)
                else:
                    label = label_int
                
                # 2. Match Tuple Structure
                if data_tuple_len == 2:
                    combined_data.append((img, label))
                elif data_tuple_len == 3:
                    # If real data returns (img, label, index), we need a dummy index
                    dummy_idx = -1 
                    combined_data.append((img, label, dummy_idx))
                else:
                    # Generic fallback for other tuple lengths
                    combined_data.append((img, label) + (0,) * (data_tuple_len - 2))
            # --- [FIX END] ---

        # Create DataLoader
        from torch.utils.data import DataLoader # Ensure this is imported
        return DataLoader(combined_data, batch_size=self.batch_size, shuffle=True)
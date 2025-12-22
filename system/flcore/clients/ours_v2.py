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
            self.img_size = 32; self.nz = 512
        elif 'imagenet' in args.dataset.lower():
            self.img_size = 64 if '100' in args.dataset else 224; self.nz = 256
        else:
            self.img_size = 32; self.nz = 100
            
        self.nc = 3
        
        # --- Hyperparameters ---
        self.replay_weight = getattr(args, 'replay_weight', 1.0)
        self.T = getattr(args, 'T', 2.0) 

        # --- Initialize Generator ---
        self.generator = Generator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=self.nc, 
            num_classes=args.num_classes, device=self.device
        ).to(self.device)
        self.freeze_generator()

        self.old_network = None # Teacher model

    def freeze_generator(self):
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def set_generator_parameters(self, global_generator):
        self.generator.load_state_dict(global_generator.state_dict())
        self.freeze_generator()

    def next_task(self, train, label_info=None, if_label=True):
        """Snapshot the current model as the Teacher before moving to the next task"""
        super().next_task(train, label_info, if_label)
        
        self.old_network = copy.deepcopy(self.model)
        self.old_network.eval()
        for param in self.old_network.parameters():
            param.requires_grad = False

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
        
        # Ensure model is trainable
        for param in self.model.parameters(): param.requires_grad = True
        self.model.train()
        self.generator.eval()
        
        start_time = time.time()
        local_loss = []
        
        for epoch in range(self.local_epochs):
            for i, (x_real, y_real) in enumerate(trainloader):
                self.optimizer.zero_grad()
                if isinstance(x_real, list): x_real = x_real[0]
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)

                # --- 1. Real Data Loss ---
                output_real = self.model(x_real)
                loss_real = self.loss(output_real, y_real)

                # --- 2. Replay Data Loss (KD) ---
                loss_replay = torch.tensor(0.0).to(self.device)
                
                if self.old_network is not None and len(self.classes_past_task) > 0:
                    batch_size = x_real.shape[0]
                    
                    z = torch.randn(batch_size, self.nz).to(self.device)
                    fake_labels_np = np.random.choice(self.classes_past_task, batch_size)
                    fake_labels = torch.tensor(fake_labels_np).long().to(self.device)
                    
                    with torch.no_grad():
                        syn_inputs = self.generator(z, fake_labels)
                        # Teacher Soft Targets
                        teacher_logits = self.old_network(syn_inputs)

                    # Student Predictions
                    student_logits = self.model(syn_inputs.detach())

                    # KL Divergence
                    log_pred_student = F.log_softmax(student_logits / self.T, dim=1)
                    pred_teacher = F.softmax(teacher_logits / self.T, dim=1)
                    kd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (self.T ** 2)
                    
                    loss_replay = self.replay_weight * kd_loss

                # --- 3. Total Loss ---
                total_loss = loss_real + loss_replay
                local_loss.append(total_loss.item())
                total_loss.backward()
                self.optimizer.step()
            
            # if self.learning_rate_scheduler:
            #     self.learning_rate_scheduler.step()
                
        # Logging
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"[Client {self.id}] Task {task} | Loss: {sum(local_loss)/len(local_loss):.4f} "
              f"(CE: {loss_real.item():.4f}, KD: {loss_replay.item():.4f}) | LR: {current_lr:.6f}")
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
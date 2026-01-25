import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flcore.clients.clientbase import Client


class GradientEncodingNet(nn.Module):
    """
    A shallow LeNet-like network used for encoding prototypes into gradients
    for the Proxy Server to reconstruct.
    """
    def __init__(self, num_classes):
        super(GradientEncodingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class GLFCClient(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        
        # GLFC specific attributes
        self.exemplar_memory = [] # Stores (image, label) tuples
        self.memory_per_class = 20 # Example size per class [cite: 269]
        self.old_model = None
        self.best_old_model = None # Received from Proxy Server
        
        # For Task Transition Detection [cite: 196]
        self.prev_entropy = 0
        self.entropy_threshold = 1.2 
        
        # Gradient Encoding Network for Prototype Communication
        self.gamma_net = GradientEncodingNet(args.num_classes).to(self.device)
        # self.gamma_net.load_state_dict(torch.load("gamma_net.pth")) # Assume pre-shared weights

    def detect_task_transition(self):
        """
        Calculates entropy of predictions to detect if new classes have arrived.
        """
        self.model.eval()
        trainloader = self.load_train_data(task=self.current_task)
        entropy_sum = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, _ in trainloader:
                x = x.to(self.device)
                outputs = F.softmax(self.model(x), dim=1)
                entropy = -torch.sum(outputs * torch.log(outputs + 1e-9), dim=1)
                entropy_sum += entropy.sum().item()
                total_samples += x.size(0)
        
        avg_entropy = entropy_sum / total_samples
        
        # Check for spike in entropy
        if (avg_entropy - self.prev_entropy) >= self.entropy_threshold:
            print(f"Client {self.id}: Task Transition Detected!")
            self.prev_entropy = avg_entropy
            return True
        
        self.prev_entropy = avg_entropy
        return False

    def generate_prototype_gradients(self):
        """
        Selects prototypes, perturbs them, and computes gradients via Gamma Net.
        """
        # 1. Select Prototypes (Closest to mean)
        # Simplified: Just picking random samples for this snippet, 
        # normally you compute class means and find closest sample.
        prototypes = []
        labels = []
        
        # Assume self.current_labels contains new classes for this task
        trainloader = self.load_train_data(task=self.current_task, batch_size=1)
        seen_classes = set()
        
        for x, y in trainloader:
            label = y.item()
            if label in self.current_labels and label not in seen_classes:
                prototypes.append(x)
                labels.append(y)
                seen_classes.add(label)
        
        if not prototypes:
            return None

        prototypes = torch.cat(prototypes).to(self.device)
        labels = torch.cat(labels).to(self.device)

        # 2. Perturb Prototypes [cite: 241, 243]
        # (Simplified perturbation logic)
        noise = torch.randn_like(prototypes) * 0.1
        perturbed_prototypes = prototypes + noise
        
        # 3. Compute Gradients via Gamma Net
        perturbed_prototypes.requires_grad = True
        outputs = self.gamma_net(perturbed_prototypes)
        loss = self.loss(outputs, labels)
        
        # Get gradients of the Gamma Net parameters
        gradients = torch.autograd.grad(loss, self.gamma_net.parameters())
        
        return gradients

    def update_memory(self):
        """Standard reservoir sampling or similar to update exemplar memory"""
        # Implementation omitted for brevity, standard CIL practice
        pass

    def train(self):
        """
        Local training with L_GC and L_RD.
        """
        # 1. Check for Task Transition
        if self.detect_task_transition():
            # If new task: generate proto-gradients to send to server
            proto_grads = self.generate_prototype_gradients()
            
            # Update memory with old data before loading new
            self.update_memory()
            
            # Note: In a real flow, proto_grads are sent to server, 
            # and server responds with 'best_old_model' BEFORE training starts.
            # Here we assume self.best_old_model is set by server via set_parameters.
        
        trainloader = self.load_train_data(task=self.current_task)
        self.model.train()
        
        # Hyperparameters for losses [cite: 185]
        lambda1 = 0.5 
        lambda2 = 0.5 if self.current_task > 0 else 0.0

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # --- Forward Pass ---
                logits = self.model(x)
                
                # --- L_GC: Class-Aware Gradient Compensation [cite: 112, 169] ---
                # We need gradients per sample to re-weight. 
                # Approximation: Reweight loss based on class type (New vs Old)
                
                # Identify new vs old classes in this batch
                is_new_class = torch.tensor([label.item() in self.current_labels for label in y], device=self.device)
                
                # Calculate Cross Entropy (reduction='none' to weight individually)
                ce_loss = F.cross_entropy(logits, y, reduction='none')
                
                # Calculate weights (G_n and G_o)
                # Note: Exact calculation requires gradients. 
                # Here we implement the conceptual weighting:
                # Weight = |Grad| / Mean_Grad. 
                # We simply assign higher weight to new classes if they are underrepresented
                # or normalize as per Eq (4).
                
                # Simplified implementation of Eq (4) logic:
                # Assume G_new and G_old are computed or estimated. 
                # For this snippet, we use scalar placeholders.
                w_new = 1.0 
                w_old = 1.0 # These would be dynamic in full implementation
                
                weights = torch.where(is_new_class, w_new, w_old)
                l_gc = (ce_loss * weights).mean()

                # --- L_RD: Class-Semantic Relation Distillation [cite: 171, 179] ---
                l_rd = 0
                if self.best_old_model is not None:
                    with torch.no_grad():
                        old_logits = self.best_old_model(x)
                        old_probs = F.softmax(old_logits, dim=1)
                    
                    # Create "Softened" targets Y_l^t [cite: 177]
                    # Current logits for new classes, Old probs for old classes
                    # (Simplified KL Divergence implementation)
                    log_probs = F.log_softmax(logits, dim=1)
                    
                    # We match the old model's predictions on the old class indices
                    # This requires mapping indices, assuming 0..C_old are old classes
                    num_old = old_logits.shape[1]
                    l_rd = F.kl_div(log_probs[:, :num_old], old_probs, reduction='batchmean')

                # --- Total Loss ---
                loss = lambda1 * l_gc + lambda2 * l_rd
                
                loss.backward()
                self.optimizer.step()
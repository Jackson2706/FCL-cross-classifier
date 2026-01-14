import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from flcore.clients.clientbase import Client

class FedLwFClient(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        self.teacher_model = None
        self.lambda_dist = 1.0  # Balance weight (lambda_o in LwF paper)
        self.T = 2.0            # Temperature, T=2 recommended by LwF paper

    def train(self, task):
        """
        FedLwF Client Training Loop.
        """
        trainloader = self.load_train_data(task=task)
        self.model.to(self.device)
        self.model.train()

        # Setup Teacher for LwF (Knowledge Distillation)
        # Teacher is the Global Model from the END of the previous task
        if self.current_task > 0 and self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            
            # Freeze teacher to save memory/compute
            for param in self.teacher_model.parameters():
                param.requires_grad = False

        # Calculate number of old classes
        # Assuming args.num_classes is total and tasks are evenly split
        # Or derived from model output size
        classes_per_task = self.args.cpt
        num_old_classes = self.current_task * classes_per_task

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass student
                logits = self.model(x)
                
                # --- 1. New Task Loss (Cross Entropy) ---
                # LwF paper Eq (1): L_new = - y_n * log(y_hat_n)
                # Standard CE handles this. 
                # Note: In strict Class-IL, we mask logits or let CE handle the full vector 
                # if labels are global indices.
                loss_ce = self.loss(logits, y)

                loss_kd = torch.tensor(0., device=self.device)

                # --- 2. Old Task Loss (Knowledge Distillation) ---
                # LwF paper Eq (2): L_old(Y_o, Y_hat_o)
                if self.current_task > 0 and self.teacher_model is not None:
                    with torch.no_grad():
                        # Get recorded responses Y_o from original network
                        teacher_logits = self.teacher_model(x)
                    
                    # We only distill knowledge for OLD classes
                    # Slice logits to [:, :num_old_classes]
                    # Note: teacher_logits might only have num_old_classes outputs 
                    # if the architecture grew, or same size if pre-allocated.
                    # We assume teacher_logits matches the size of old classes.
                    
                    # Check dimensions to be safe
                    teacher_out_dim = teacher_logits.shape[1]
                    student_old_logits = logits[:, :teacher_out_dim]
                    
                    # LwF Distillation Loss (Modified Cross Entropy with Temperature)
                    # Formula Eq (4): Softmax with T
                    
                    # Log-Softmax on Student (Old Classes)
                    student_log_probs = F.log_softmax(student_old_logits / self.T, dim=1)
                    
                    # Softmax on Teacher
                    teacher_probs = F.softmax(teacher_logits / self.T, dim=1)
                    
                    # KL Div Loss (equivalent to cross-entropy for soft targets up to a constant)
                    # PyTorch KLDivLoss expects input as LogProb
                    # We multiply by T^2 as per Hinton's Distillation paper guidance 
                    # (referenced in LwF paper) to keep gradients invariant to T
                    loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.T ** 2)

                # Total Loss
                # LwF Paper Fig 3: lambda * L_old + L_new + R
                loss = loss_ce + self.lambda_dist * loss_kd
                
                loss.backward()
                self.optimizer.step()
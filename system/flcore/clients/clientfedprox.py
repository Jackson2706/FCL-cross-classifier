import copy
import time

import numpy as np
import torch
from flcore.clients.clientbase import Client


class clientProx(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        
        # Get the proximal coefficient (mu) from args
        self.mu = args.mu 

    def train(self, task=None):
        trainloader = self.load_train_data(task=task)
        self.model.train()
        
        # 1. Deep copy the global model to use as the fixed reference (w^t)
        # self.model currently holds the global weights at the start of the round
        global_model = copy.deepcopy(self.model)
        
        # Disable gradient calculation for global_model to save memory
        for param in global_model.parameters():
            param.requires_grad = False
        
        start_time = time.time()

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                
                # 2. Calculate original loss
                loss = self.loss(output, y)

                # 3. Add Proximal Term: (mu / 2) * ||w - w^t||^2
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)**2

                loss += (self.mu / 2) * proximal_term

                loss.backward()
                self.optimizer.step()
            
            # Step the scheduler if it exists
            if self.learning_rate_scheduler:
                self.learning_rate_scheduler.step()
        
        # Cleanup to free GPU memory
        del global_model

        if self.args.teval:
            self.grad_eval(old_model=self.model)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
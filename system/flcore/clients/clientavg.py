import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.clients.clientbase import Client

class clientAVG(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

    def train(self, task=None):
        """
        Train model cục bộ trên dữ liệu của task hiện tại.
        """
        print(f"[Client-side] ID: {self.id}, task-id: {task}")
        trainloader = self.load_train_data(task=task)
        self.model.train()
        
        start_time = time.time()
        max_local_epochs = self.local_epochs

        # Vòng lặp training cục bộ
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Cập nhật learning rate scheduler nếu có
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        # Ghi nhận chi phí thời gian
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
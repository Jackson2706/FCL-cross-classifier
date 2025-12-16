import torch
import torch.nn as nn
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from flcore.utils_core.target_utils import Generator

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
        self.generator = Generator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=self.nc, device=self.device
        ).to(self.device)
        
        # Luôn đóng băng Generator ở Client
        self.freeze_generator()

        # Tham số Replay
        self.replay_weight = args.replay_weight if hasattr(args, 'replay_weight') else 1.0

    def freeze_generator(self):
        """Đảm bảo Generator không bao giờ được train ở Client"""
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def set_generator_parameters(self, global_generator):
        """
        Nhận Generator mới nhất từ Server
        """
        self.generator.load_state_dict(global_generator.state_dict())
        self.freeze_generator() # Đảm bảo vẫn đóng băng sau khi load

    def train(self, task=None):
        trainloader = self.load_train_data(task=task)
        self.model.train()
        self.generator.eval()
        
        start_time = time.time()
        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x_real, y_real) in enumerate(trainloader):
                # --- A. Xử lý dữ liệu thật (Real Data) ---
                if isinstance(x_real, list):
                    x_real = x_real[0].to(self.device)
                else:
                    x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)

                # --- B. Xử lý dữ liệu giả (Replay Data) ---
                loss_replay = 0
                
                # Điều kiện: Có task cũ VÀ Generator đã học được gì đó (task > 0)
                if self.current_task > 0 and len(self.classes_past_task) > 0:
                    batch_size_replay = x_real.shape[0]
                    
                    # 1. Sinh vector nhiễu (z) và chọn nhãn cũ ngẫu nhiên
                    z = torch.randn(batch_size_replay, self.nz).to(self.device)
                    fake_labels = np.random.choice(self.classes_past_task, batch_size_replay)
                    fake_labels = torch.tensor(fake_labels).long().to(self.device)
                    
                    # 2. Sinh ảnh từ Generator (No gradient)
                    with torch.no_grad():
                        x_fake = self.generator(z, fake_labels)
                    
                    # 3. Tính Loss Replay (Classifier phải nhận diện đúng ảnh giả)
                    output_fake = self.model(x_fake.detach()) 
                    loss_replay = self.loss(output_fake, fake_labels)

                # --- C. Tổng hợp và Update ---
                output_real = self.model(x_real)
                loss_real = self.loss(output_real, y_real)

                total_loss = loss_real + self.replay_weight * loss_replay
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
import time
import torch
import copy
import numpy as np
from torch import nn, optim
from tqdm import tqdm

from flcore.servers.serverbase import Server
from flcore.clients.ours import clientOurs
from flcore.utils_core.target_utils import Generator 

# Hàm khởi tạo trọng số Kaiming He
def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Ours(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # --- 1. Cấu hình Generator ---
        # Đảm bảo logic giống hệt Client để match kích thước
        if 'cifar' in self.dataset.lower():
            self.img_size = 32
            self.nz = 256
        elif 'imagenet' in self.dataset.lower():
            self.img_size = 64 if '100' in self.dataset else 224
            self.nz = 256
        else:
            self.img_size = 64
            self.nz = 100

        # Khởi tạo Global Generator
        self.global_generator = Generator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=3, device=self.device
        ).to(self.device)

        # Cờ kiểm soát việc khởi tạo trọng số (chỉ init 1 lần)
        self.has_initialized_generator = False 

        # Hyperparameters cho Generator Training
        self.g_lr = args.g_lr if hasattr(args, 'g_lr') else 0.001
        self.g_steps = args.g_steps if hasattr(args, 'g_steps') else 100 
        self.batch_size_gen = 64
        
        self.set_clients(clientOurs)
        print(f"Server Initialized. Generator Config: nz={self.nz}, size={self.img_size}")

    def train(self):
        for task in range(self.args.num_tasks):
            print(f"\n================ Current Task: {task} =================")
            self.current_task = task
            
            # Update thông tin nhãn cho task mới
            self._update_label_info()
            
            # Load dữ liệu cho Client (trừ task 0 đã load sẵn trong init)
            if task > 0:
                self._load_task_data_for_clients(task)

            # --- Vòng lặp Round ---
            for i in range(self.global_rounds):
                s_t = time.time()
                
                # 1. Select Clients
                self.selected_clients = self.select_clients()
                
                # 2. Send Models (Gửi Classifier + Generator)
                self.send_models() 

                # 3. Local Training (Client train Classifier)
                # Client chỉ dùng Generator để replay, không train nó
                for client in self.selected_clients:
                    client.train(task=task)

                # 4. Receive Models (Chỉ nhận về Classifier)
                self.receive_models() 
                
                # 5. Server-side Generator Training
                # Chỉ train Generator nếu đã có models gửi về
                if len(self.uploaded_models) > 0:
                    print(f"\n[Round {i}] Training Global Generator on Server...")
                    self.train_global_generator(available_labels=self.available_labels_current)
                
                # 6. Aggregate Classifiers (FedAvg)
                self.aggregate_parameters()

                self.Budget.append(time.time() - s_t)
                
                # 7. Evaluation
                if i % self.eval_gap == 0:
                    self.eval(task=task, glob_iter=i + self.global_rounds * task, flag="global")

    def train_global_generator(self, available_labels):
        """
        Train Generator để thỏa mãn Ensemble các Classifiers của Clients.
        """
        available_labels_list = list(available_labels)
        if len(available_labels_list) == 0: return

        # --- A. Kiểm tra khởi tạo (Logic quan trọng) ---
        if not self.has_initialized_generator:
            print(">>> [First Time] Initializing Generator Weights (Kaiming He)...")
            self.global_generator.apply(weight_init)
            self.has_initialized_generator = True
        else:
            # Các lần sau không init lại -> Generator tiếp tục học từ kiến thức cũ
            # print(">>> [Continue] Fine-tuning Generator...") 
            pass

        self.global_generator.train()

        # --- B. Chuẩn bị Teachers (Classifiers từ Clients) ---
        teachers = self.uploaded_models
        for t in teachers:
            t.eval()
            for p in t.parameters():
                p.requires_grad = False # Freeze teachers

        # --- C. Setup Optimizer ---
        # Tạo mới optimizer để reset momentum, nhưng trọng số Generator vẫn giữ nguyên
        optimizer_g = optim.Adam(self.global_generator.parameters(), lr=self.g_lr)
        criterion = nn.CrossEntropyLoss()

        # --- D. Training Loop ---
        # Dùng tqdm để hiện thanh tiến trình nếu muốn, hoặc bỏ đi cho gọn log
        # pbar = tqdm(range(self.g_steps), desc="Optimizing Generator", leave=False)
        
        for step in range(self.g_steps):
            # 1. Sample Noise & Labels
            labels = np.random.choice(available_labels_list, self.batch_size_gen)
            labels = torch.tensor(labels).long().to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)

            # 2. Generate Images
            gen_imgs = self.global_generator(z, labels)

            # 3. Calculate Ensemble Loss
            # Loss là trung bình cộng CrossEntropy của tất cả teacher trên ảnh sinh ra
            total_loss = 0
            for teacher in teachers:
                preds = teacher(gen_imgs)
                total_loss += criterion(preds, labels)
            
            loss_g = total_loss / len(teachers)

            # 4. Update Generator
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

    def send_models(self):
        """
        Gửi Global Model (Classifier) và Global Generator xuống Clients
        """
        assert (len(self.clients) > 0)

        for client in self.clients:
            # 1. Gửi Classifier (cho việc train tiếp theo)
            client.set_parameters(self.global_model)
            
            # 2. Gửi Generator (cho việc Replay dữ liệu cũ)
            client.set_generator_parameters(self.global_generator)

    def _update_label_info(self):
        """Cập nhật tập hợp tất cả các nhãn hiện có trong hệ thống"""
        available_labels = set()
        for u in self.clients:
            available_labels = available_labels.union(set(u.classes_so_far))
        self.available_labels_current = available_labels

    def _load_task_data_for_clients(self, task):
        """Helper để load dữ liệu task mới cho từng client"""
        for i, client in enumerate(self.clients):
            # Logic chọn hàm đọc dữ liệu
            if self.args.dataset == 'IMAGENET1k':
                from utils.data_utils import read_client_data_FCL_imagenet1k
                read_func = read_client_data_FCL_imagenet1k
            elif 'CIFAR100' in self.args.dataset:
                from utils.data_utils import read_client_data_FCL_cifar100
                read_func = read_client_data_FCL_cifar100
            else: # CIFAR10
                from utils.data_utils import read_client_data_FCL_cifar10
                read_func = read_client_data_FCL_cifar10
            
            train_data, label_info = read_func(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
            
            # Gọi next_task (Client tự động freeze model cũ bên trong)
            client.next_task(train_data, label_info)
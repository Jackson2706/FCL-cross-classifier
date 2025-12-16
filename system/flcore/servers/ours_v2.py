import time
import torch
import torch.nn.functional as F
import copy
import numpy as np
from torch import nn, optim
from tqdm import tqdm

from flcore.servers.serverbase import Server
from flcore.clients.ours_v2 import clientOursV2
from flcore.utils_core.target_utils import Generator 

def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class OursV2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # --- Config Generator ---
        if 'cifar' in self.dataset.lower():
            self.img_size = 32; self.nz = 256
        elif 'imagenet' in self.dataset.lower():
            self.img_size = 64 if '100' in self.dataset else 224; self.nz = 256
        else:
            self.img_size = 64; self.nz = 100

        self.global_generator = Generator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=3, device=self.device
        ).to(self.device)

        self.has_initialized_generator = False 
        
        # Hyperparameters
        self.g_lr = getattr(args, 'g_lr', 0.001)     # LR cho Generator
        self.c_lr = getattr(args, 'c_lr', 0.001)     # LR cho Classifier (Server training)
        self.g_steps = getattr(args, 'g_steps', 100) # Steps train Generator
        self.k_steps = getattr(args, 'k_steps', 100) # Steps train Classifier (KD)
        self.batch_size_gen = 64
        self.T = getattr(args, 'T', 2.0)             # Temperature cho KD
        
        self.set_clients(clientOursV2)
        print(f"Server Initialized (Ours + Server KD). Generator Config: nz={self.nz}")

    def train(self):
        for task in range(self.args.num_tasks):
            print(f"\n================ Current Task: {task} =================")
            self.current_task = task
            self._update_label_info()
            
            if task > 0: self._load_task_data_for_clients(task)

            for i in range(self.global_rounds):
                s_t = time.time()
                glob_iter = i + self.global_rounds * task
                
                # 1. Send Models
                self.selected_clients = self.select_clients()
                self.send_models() 

                # 2. Local Training
                for client in self.selected_clients:
                    client.train(task=task)

                # 3. Receive Client Models
                self.receive_models() 
                
                if len(self.uploaded_models) > 0:
                    # A. Train Generator (GAN Loss: Fooling Ensemble)
                    print(f"\n[Round {i}] Training Global Generator...")
                    self.train_global_generator(self.available_labels_current)

                    # B. Aggregate (FedAvg initialization)
                    # Chúng ta vẫn cần FedAvg để lấy điểm khởi đầu tốt cho Global Model
                    self.aggregate_parameters()

                    # C. Train Global Classifier (KL Loss: Learning from Ensemble)
                    print(f"[Round {i}] Distilling Ensemble into Global Classifier...")
                    self.train_global_classifier(self.available_labels_current)
                else:
                    self.aggregate_parameters()
                
                self.Budget.append(time.time() - s_t)
                
                # Evaluation
                if i % self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

    def train_global_generator(self, available_labels):
        """
        Train Generator để sinh ảnh mà Ensemble classifiers đều tự tin phân loại.
        Loss: CrossEntropy (Adversarial Loss).
        """
        labels_list = list(available_labels)
        if len(labels_list) == 0: return

        if not self.has_initialized_generator:
            print(">>> [Init] Generator Weights...")
            self.global_generator.apply(weight_init)
            self.has_initialized_generator = True
        
        self.global_generator.train()
        
        # Freeze Teachers (Ensemble)
        teachers = self.uploaded_models
        for t in teachers:
            t.eval()
            for p in t.parameters(): p.requires_grad = False

        optimizer_g = optim.Adam(self.global_generator.parameters(), lr=self.g_lr)
        criterion_gan = nn.CrossEntropyLoss()

        for step in range(self.g_steps):
            # Sample Z & Labels
            labels = np.random.choice(labels_list, self.batch_size_gen)
            labels = torch.tensor(labels).long().to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)

            gen_imgs = self.global_generator(z, labels)

            # Ensemble Loss
            total_loss = 0
            for teacher in teachers:
                preds = teacher(gen_imgs)
                total_loss += criterion_gan(preds, labels)
            
            loss_g = total_loss / len(teachers)

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

    def train_global_classifier(self, available_labels):
        """
        Train Global Classifier (Student) để bắt chước Ensemble (Teachers) 
        sử dụng dữ liệu sinh từ Generator.
        Loss: KL Divergence.
        """
        labels_list = list(available_labels)
        if len(labels_list) == 0: return

        self.global_model.train() # Student cần được train
        self.global_generator.eval() # Generator dùng để sinh, không train ở bước này

        # Freeze Teachers (Ensemble)
        teachers = self.uploaded_models
        for t in teachers:
            t.eval()
            for p in t.parameters(): p.requires_grad = False
            
        # Optimizer cho Global Classifier
        # Lưu ý: Chỉ optimize classifier, không đụng vào generator
        optimizer_c = optim.Adam(self.global_model.parameters(), lr=self.c_lr)
        criterion_kd = nn.KLDivLoss(reduction='batchmean')

        for step in range(self.k_steps):
            # 1. Sinh dữ liệu giả (không cần label cụ thể vì ta học theo teacher)
            # Tuy nhiên Generator là conditional nên vẫn cần label input
            labels = np.random.choice(labels_list, self.batch_size_gen)
            labels = torch.tensor(labels).long().to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)

            with torch.no_grad():
                gen_imgs = self.global_generator(z, labels)

            # 2. Tính Teacher Output (Ensemble Average Logits)
            teacher_logits_sum = 0
            with torch.no_grad():
                for teacher in teachers:
                    teacher_logits_sum += teacher(gen_imgs)
            
            # Trung bình cộng logits của các teacher
            teacher_avg_logits = teacher_logits_sum / len(teachers)

            # 3. Tính Student Output
            student_logits = self.global_model(gen_imgs)

            # 4. KL Loss
            # Input của KLDivLoss cần là LogSoftmax
            # Target của KLDivLoss cần là Softmax (probabilities)
            loss_kd = criterion_kd(
                F.log_softmax(student_logits / self.T, dim=1),
                F.softmax(teacher_avg_logits / self.T, dim=1)
            ) * (self.T * self.T)

            # 5. Update Global Model
            optimizer_c.zero_grad()
            loss_kd.backward()
            optimizer_c.step()

    def send_models(self):
        for client in self.clients:
            client.set_parameters(self.global_model)
            client.set_generator_parameters(self.global_generator)

    def _update_label_info(self):
        available_labels = set()
        for u in self.clients:
            available_labels = available_labels.union(set(u.classes_so_far))
        self.available_labels_current = available_labels

    def _load_task_data_for_clients(self, task):
        # Import data utils logic
        if self.args.dataset == 'IMAGENET1k':
            from utils.data_utils import read_client_data_FCL_imagenet1k as read_func
        elif 'CIFAR100' in self.args.dataset:
            from utils.data_utils import read_client_data_FCL_cifar100 as read_func
        else:
            from utils.data_utils import read_client_data_FCL_cifar10 as read_func
        
        for i, client in enumerate(self.clients):
            train_data, label_info = read_func(
                i, task=task, classes_per_task=self.args.cpt, count_labels=True
            )
            client.next_task(train_data, label_info)
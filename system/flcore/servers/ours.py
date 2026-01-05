import time
import torch
import torch.nn.functional as F
import copy
import numpy as np
import os
from torch import nn, optim
from torchvision.utils import save_image
from flcore.servers.serverbase import Server
from flcore.clients.ours import clientOurs # Đảm bảo import đúng file client
from flcore.utils_core.target_utils import *
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k, read_client_data_FCL_cifar10

# ==========================================
# 1. ADVANCED GENERATOR (Phải khớp với Client)
# ==========================================
class AdvancedGenerator(nn.Module):
    """Generator optimized for 224x224 images with better class conditioning"""
    def __init__(self, nz=100, ngf=128, img_size=224, nc=3, num_classes=7):
        super(AdvancedGenerator, self).__init__()
        self.params = (nz, ngf, img_size, nc, num_classes)
        self.num_classes = num_classes
        
        # Xác định init_size dựa trên img_size
        if img_size == 224: self.init_size = 7
        elif img_size == 64: self.init_size = 4
        elif img_size == 32: self.init_size = 4
        else: self.init_size = img_size // 32
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        input_dim = nz + num_classes
        
        # Initial projection
        # Note: Logic này cần khớp với Client. Ở đây tôi dùng bản simplified để demo,
        # Nếu Client dùng logic tính num_stages động, hãy copy y nguyên Class đó vào đây.
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, ngf * 16 * self.init_size ** 2),
            nn.BatchNorm1d(ngf * 16 * self.init_size ** 2),
            nn.ReLU(True)
        )

        self.conv_blocks = nn.Sequential(
            self._upsample_block(ngf*16, ngf*8), # 7->14 (or 4->8)
            self._upsample_block(ngf*8, ngf*8),  # 14->28 (or 8->16)
            self._upsample_block(ngf*8, ngf*4),  # 28->56 (or 16->32)
            # Logic xử lý kích thước ảnh nhỏ (32x32) hay lớn (224x224)
            # Lưu ý: Cấu trúc dưới đây giả định cho 224.
            # Với CIFAR (32), Generator cần điều chỉnh số lớp. 
            # Để an toàn, tôi giữ nguyên cấu trúc bạn cung cấp.
            self._upsample_block(ngf*4, ngf*2),
            self._upsample_block(ngf*2, ngf),
            
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Sigmoid(), 
        )

        # Xử lý riêng cho CIFAR (32x32) để tránh lỗi dimension nếu dùng code của 224
        if img_size == 32:
             self.conv_blocks = nn.Sequential(
                self._upsample_block(ngf*16, ngf*8), # 4->8
                self._upsample_block(ngf*8, ngf*4),  # 8->16
                self._upsample_block(ngf*4, ngf*2),  # 16->32
                nn.Conv2d(ngf*2, ngf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                nn.Conv2d(ngf, nc, 3, 1, 1),
                nn.Sigmoid(),
            )

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, z, labels):
        batch_size = z.size(0)
        label_emb = self.label_emb(labels)
        gen_input = torch.cat([z, label_emb], dim=1)
        out = self.l1(gen_input)
        # Reshape linh hoạt dựa trên output của linear layer
        out = out.view(batch_size, -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
class BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_feature = None 

    def hook_fn(self, module, input, output):
        self.r_feature = input[0]

    def remove(self):
        self.hook.remove()

def get_bn_loss(teacher_model, gen_imgs_normalized):
    """Calculates BN statistics distance using NORMALIZED images."""
    bn_hooks = []
    bn_layers = [m for m in teacher_model.modules() if isinstance(m, nn.BatchNorm2d)]

    for module in bn_layers:
        bn_hooks.append(BNFeatureHook(module))

    teacher_model(gen_imgs_normalized)

    loss_bn = 0.0
    for hook, layer in zip(bn_hooks, bn_layers):
        real_mean = layer.running_mean
        real_var = layer.running_var
        gen_feat = hook.r_feature
        # Tính stat của feature map hiện tại
        gen_mean = torch.mean(gen_feat, dim=[0, 2, 3])
        gen_var = torch.var(gen_feat, dim=[0, 2, 3], unbiased=False)
        loss_bn += torch.norm(gen_mean - real_mean, 2) + torch.norm(gen_var - real_var, 2)

    for hook in bn_hooks: hook.remove()
    return loss_bn

def KD_loss(logits, targets, T=2.0):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction='batchmean') * (T * T)

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.constant_(m.bias, 0)

# ==========================================
# 3. SERVER CLASS
# ==========================================
class Ours(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.Budget = []
        
        # --- Config Dataset Stats (QUAN TRỌNG: Để chuẩn hóa ảnh giả) ---
        if 'cifar100' in self.args.dataset.lower():
            self.img_size = 32; self.nz = 256
            self.mean = [0.5071, 0.4867, 0.4408]
            self.std = [0.2675, 0.2565, 0.2761]
        elif 'cifar10' in self.args.dataset.lower():
            self.img_size = 32; self.nz = 100
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
        elif 'imagenet' in self.args.dataset.lower():
            self.img_size = 64 if '100' in self.args.dataset else 224; self.nz = 256
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.img_size = 32; self.nz = 100
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]

        # --- Init Generator ---
        self.global_generator = AdvancedGenerator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=3, 
            num_classes=args.num_classes
        ).to(self.device)
        self.global_generator.apply(weight_init)

        self.prev_generator = None
        
        # Hyperparameters
        self.g_lr = getattr(args, 'g_lr', 0.0002)
        self.c_lr = getattr(args, 'c_lr', 0.001)     
        self.g_steps = getattr(args, 'g_steps', 100) 
        self.k_steps = getattr(args, 'k_steps', 200) 
        self.batch_size_gen = 32 # Tăng lên một chút nếu GPU cho phép
        self.T = getattr(args, 'T', 2.0)        

        self.optimizer_g = optim.Adam(self.global_generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        self.optimizer_c = optim.Adam(self.global_model.parameters(), lr=self.c_lr)

        self.scheduler_g = optim.lr_scheduler.MultiStepLR(self.optimizer_g, milestones=[30, 80], gamma=0.1)
        self.scheduler_c = optim.lr_scheduler.MultiStepLR(self.optimizer_c, milestones=[30, 80], gamma=0.1)

        self.set_clients(clientOurs)
        print(f"Server Initialized. Generator Config: ImgSize={self.img_size}, nz={self.nz}")

    def batch_normalize(self, batch_tensor, mean, std):
        """Chuẩn hóa batch ảnh [0,1] về distribution của dataset thật."""
        mean_t = torch.tensor(mean, device=batch_tensor.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, device=batch_tensor.device).view(1, -1, 1, 1)
        return (batch_tensor - mean_t) / std_t

    def train(self):
        for task in range(self.args.num_tasks):
            print(f"\n================ Current Task: {task} =================")
            
            # --- 1. Label Info & Data Setup ---
            if task == 0:
                # Task đầu tiên: chưa có task cũ
                available_labels = set()
                available_labels_current = set()
                for u in self.clients:
                    available_labels.update(u.classes_so_far)
                    available_labels_current.update(u.current_labels)
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = [] # Task 0 không có quá khứ
            else:
                self.current_task = task
                torch.cuda.empty_cache()
                # Load dữ liệu mới cho task hiện tại
                for i in range(len(self.clients)):
                    if self.args.dataset == 'IMAGENET1k':
                        read_func = read_client_data_FCL_imagenet1k
                    elif 'cifar100' in self.args.dataset.lower():
                        read_func = read_client_data_FCL_cifar100
                    else:
                        read_func = read_client_data_FCL_cifar10
                    
                    train_data, label_info = read_func(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    self.clients[i].next_task(train_data, label_info)

                # Update labels info
                available_labels = set()
                available_labels_current = set()
                
                # Gom tất cả label hiện có
                for u in self.clients:
                    available_labels.update(u.classes_so_far)
                    available_labels_current.update(u.current_labels)
                
                # Xác định nhãn quá khứ (Past Labels) = Tất cả - Hiện tại
                # Cách này chính xác hơn việc lấy từ client[0]
                available_labels_past = available_labels - available_labels_current

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past) # Client sẽ dùng list này để Replay

            # --- 2. Training Rounds ---
            for i in range(self.global_rounds):
                s_t = time.time()
                glob_iter = i + self.global_rounds * task
                
                self.selected_clients = self.select_clients() 
                
                # [FIXED] Phải gửi model  xuống client!
                self.send_models(send_generator=(task>0))
                
                # Eval trước khi train (tuỳ chọn)
                if i % 5 == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                print(f"\n-------------Round number: {i}-------------")
                for client in self.selected_clients:
                    client.train(task=task)
                
                # Nhận model để aggregate
                self.receive_models()
                self.aggregate_parameters()
                
                self.Budget.append(time.time() - s_t)
                print(f"Round {i} Time: {self.Budget[-1]:.2f}s")

            # --- 3. Post-Task Processing ---
            # Sau khi xong task, huấn luyện Generator và Global Model để chuẩn bị cho task sau
            self.receive_models()
            self.train_global_generator()
            self.train_global_classifier()
            
            # --- 4. Evaluation & Debug ---
            print(f"\n>>> [Eval] Detailed Evaluation for Task {task} (Real vs Synthetic)")
            self.eval_task(task=task, glob_iter=task, flag="global")

    def train_global_generator(self):
        print(f"[Server] Start training Generator (with BN Reg)")
        
        # Chuẩn bị dữ liệu label từ các client
        processed_client_labels = {}
        all_labels_set = set()

        for client_id, info in self.client_info_dict.items():
            # Xử lý format label (tensor/list/numpy) về list int chuẩn
            raw_labels = info["label"]
            if isinstance(raw_labels, torch.Tensor):
                clean_labels = raw_labels.cpu().numpy().astype(int)
            elif isinstance(raw_labels, (list, set)):
                temp = list(raw_labels)
                if len(temp) > 0 and isinstance(temp[0], torch.Tensor):
                    clean_labels = np.array([x.item() for x in temp], dtype=int)
                else:
                    clean_labels = np.array(temp, dtype=int)
            else:
                clean_labels = np.array(list(raw_labels), dtype=int)
            
            processed_client_labels[client_id] = clean_labels
            all_labels_set.update(clean_labels.tolist())
            
            # Đóng băng teacher model
            info["model"].eval()
            info["model"].to(self.device)
            for param in info["model"].parameters(): param.requires_grad = False

        labels_list = list(all_labels_set)
        if len(labels_list) == 0: return

        self.global_generator.train()
        for param in self.global_generator.parameters(): param.requires_grad = True

        criterion_ce = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss() 
        alpha_anchor = 10.0
        beta_bn = 1.0 # Tăng trọng số BN loss lên để ảnh giả có thống kê giống ảnh thật

        for step in range(self.g_steps):
            self.optimizer_g.zero_grad()
            
            # A. Generate Raw Images [0, 1]
            selected_labels = np.random.choice(labels_list, self.batch_size_gen)
            labels_tensor = torch.tensor(selected_labels, dtype=torch.long).to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)
            gen_imgs_raw = self.global_generator(z, labels_tensor)

            # [FIXED] Normalize ảnh giả trước khi đưa vào Teacher Models
            gen_imgs_norm = self.batch_normalize(gen_imgs_raw, self.mean, self.std)

            # B. Teacher Losses
            total_ce_loss = torch.tensor(0.0, device=self.device)
            total_bn_loss = torch.tensor(0.0, device=self.device)
            valid_teachers = 0
            
            for client_id, info in self.client_info_dict.items():
                teacher_model = info["model"]
                teacher_labels = processed_client_labels[client_id]
                
                # Tìm các ảnh trong batch thuộc về teacher này
                mask = np.isin(selected_labels, teacher_labels)
                if mask.sum() > 0:
                    valid_teachers += 1
                    mask_tensor = torch.tensor(mask, device=self.device)
                    
                    # Lấy ảnh đã normalize để tính Loss
                    relevant_imgs_norm = gen_imgs_norm[mask_tensor]
                    relevant_labels = labels_tensor[mask_tensor]
                    
                    preds = teacher_model(relevant_imgs_norm)
                    
                    total_ce_loss += criterion_ce(preds, relevant_labels)
                    # BN Loss cũng cần ảnh normalize để match statistics
                    total_bn_loss += get_bn_loss(teacher_model, relevant_imgs_norm)

            loss_g = total_ce_loss + (beta_bn * total_bn_loss)
            
            # C. Anchor Loss (Tránh quên kiến thức của Generator cũ)
            if self.prev_generator is not None:
                self.prev_generator.eval()
                with torch.no_grad():
                    anchor_imgs_raw = self.prev_generator(z, labels_tensor)
                # So sánh ảnh raw với nhau (MSE)
                loss_g += alpha_anchor * mse_loss(gen_imgs_raw, anchor_imgs_raw)

            if valid_teachers > 0:
                loss_g.backward()
                self.optimizer_g.step()
        
        self.scheduler_g.step()
        # Lưu lại Generator hiện tại làm Anchor cho vòng sau
        self.prev_generator = copy.deepcopy(self.global_generator)

    def train_global_classifier(self, steps=100):
        print(f"[Server] Start training Global Classifier (Distillation)")
        all_labels = set()
        for v in self.client_info_dict.values():
            # Gom lại label logic
            raw_labels = v["label"]
            if isinstance(raw_labels, (np.ndarray, list)):
                 all_labels.update(list(raw_labels))
            elif isinstance(raw_labels, torch.Tensor):
                 all_labels.update(raw_labels.cpu().tolist())

        labels_list = list(all_labels)
        if len(labels_list) == 0: return

        self.global_generator.eval()
        self.global_model.train()
        for param in self.global_model.parameters(): param.requires_grad = True
        
        for step in range(steps):
            self.optimizer_c.zero_grad()
            
            # 1. Generate & Normalize
            selected_labels = np.random.choice(labels_list, self.batch_size_gen)
            labels_tensor = torch.tensor(selected_labels).long().to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)
            
            with torch.no_grad():
                gen_imgs_raw = self.global_generator(z, labels_tensor).detach()
                # [FIXED] Normalize
                gen_imgs_norm = self.batch_normalize(gen_imgs_raw, self.mean, self.std)

            # 2. Student Forward
            student_logits = self.global_model(gen_imgs_norm)

            # 3. Teachers Forward (Ensemble Logits)
            teacher_logits_sum = torch.zeros_like(student_logits)
            teacher_counts = torch.zeros(self.batch_size_gen, 1, device=self.device)
            
            for client_id, info in self.client_info_dict.items():
                mask = np.isin(selected_labels, info["label"])
                
                if mask.sum() > 0:
                    teacher_model = info["model"]
                    mask_tensor = torch.tensor(mask, device=self.device).unsqueeze(1)
                    
                    with torch.no_grad():
                        logits = teacher_model(gen_imgs_norm)
                        
                    teacher_logits_sum += logits * mask_tensor
                    teacher_counts += mask_tensor
            
            # Tránh chia cho 0
            teacher_counts[teacher_counts == 0] = 1.0
            teacher_avg_logits = teacher_logits_sum / teacher_counts
            
            loss = KD_loss(student_logits, teacher_avg_logits, self.T)
            loss.backward()
            self.optimizer_c.step()
            
        self.scheduler_c.step()

    def send_models(self, send_generator):
        print(f"[Server] Sending Global Model & Generator to {len(self.selected_clients)} clients")
        for client in self.selected_clients:
            client.set_parameters(self.global_model)
            if send_generator:
                client.set_generator_parameters(self.global_generator)

    def receive_models(self):
        print(f"[Server] Receiving models from clients...")
        self.client_info_dict = {}
        self.uploaded_models = []
        for client in self.selected_clients: # Chỉ nhận từ selected clients
            # Đảm bảo copy model để tránh tham chiếu bộ nhớ bị thay đổi
            model_copy = copy.deepcopy(client.model)
            clean_labels = list(set(client.classes_so_far))
            
            self.client_info_dict[client.id] = {
                "model": model_copy,
                "label": clean_labels
            }
            self.uploaded_models.append(model_copy)
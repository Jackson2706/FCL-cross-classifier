import time
import torch
import torch.nn.functional as F
import copy
import numpy as np
from torch import nn, optim
from flcore.servers.serverbase import Server
from flcore.clients.ours_v2 import clientOursV2
from flcore.utils_core.target_utils import *
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k, read_client_data_FCL_cifar10

# --- Helper Functions ---
class BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_feature = None 

    def hook_fn(self, module, input, output):
        self.r_feature = input[0]

    def remove(self):
        self.hook.remove()

def get_bn_loss(teacher_model, gen_imgs):
    """Calculates BN statistics distance (Mean/Var) between real (stored) and fake data."""
    bn_hooks = []
    bn_layers = [m for m in teacher_model.modules() if isinstance(m, nn.BatchNorm2d)]

    for module in bn_layers:
        bn_hooks.append(BNFeatureHook(module))

    teacher_model(gen_imgs)

    loss_bn = 0.0
    for hook, layer in zip(bn_hooks, bn_layers):
        real_mean = layer.running_mean
        real_var = layer.running_var
        gen_feat = hook.r_feature
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
# SERVER CLASS
# ==========================================
class OursV2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.Budget = []
        
        # --- Config Generator ---
        if 'cifar100' in self.dataset.lower():
            self.img_size = 32; self.nz = 512
        elif 'imagenet' in self.dataset.lower():
            self.img_size = 64 if '100' in self.dataset else 224; self.nz = 256
        else:
            self.img_size = 32; self.nz = 100

        self.global_generator = Generator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=3, 
            num_classes=args.num_classes, device=self.device
        ).to(self.device)
        self.global_generator.apply(weight_init)

        self.prev_generator = None
        
        # Hyperparameters
        self.g_lr = getattr(args, 'g_lr', 0.0002)
        self.c_lr = getattr(args, 'c_lr', 0.001)     
        self.g_steps = getattr(args, 'g_steps', 100) 
        self.k_steps = getattr(args, 'k_steps', 200) 
        self.batch_size_gen = 64
        self.T = getattr(args, 'T', 2.0)        

        self.optimizer_g = optim.Adam(self.global_generator.parameters(), lr=self.g_lr, betas=(0.5, 0.999))
        self.optimizer_c = optim.Adam(self.global_model.parameters(), lr=self.c_lr)

        self.scheduler_g = optim.lr_scheduler.MultiStepLR(self.optimizer_g, milestones=[30, 80], gamma=0.1)
        self.scheduler_c = optim.lr_scheduler.MultiStepLR(self.optimizer_c, milestones=[30, 80], gamma=0.1)

        self.set_clients(clientOursV2)
        print(f"Server Initialized. Generator Config: ImgSize={self.img_size}, nz={self.nz}")

    def train(self):
        for task in range(self.args.num_tasks):
            print(f"\n================ Current Task: {task} =================")
            
            # --- Label Info Setup ---
            if task == 0:
                available_labels = set()
                available_labels_current = set()
                for u in self.clients:
                    available_labels.update(u.classes_so_far)
                    available_labels_current.update(u.current_labels)
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = []
            else:
                self.current_task = task
                torch.cuda.empty_cache()
                for i in range(len(self.clients)):
                    if self.args.dataset == 'IMAGENET1k':
                        read_func = read_client_data_FCL_imagenet1k
                    elif 'cifar100' in self.args.dataset.lower():
                        read_func = read_client_data_FCL_cifar100
                    else:
                        read_func = read_client_data_FCL_cifar10
                    
                    train_data, label_info = read_func(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    self.clients[i].next_task(train_data, label_info)

                # Update global labels
                available_labels = set()
                available_labels_current = set()
                for u in self.clients:
                    available_labels.update(u.classes_so_far)
                    available_labels_current.update(u.current_labels)
                
                # Get past labels from one client (since they are consistent now)
                available_labels_past = self.clients[0].classes_past_task 

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            # --- Training Rounds ---
            for i in range(self.global_rounds):
                print(f"\n-------------Round number: {i}-------------")
                s_t = time.time()
                glob_iter = i + self.global_rounds * task
                
                self.selected_clients = self.select_clients() 
                self.eval(task=task, glob_iter=glob_iter, flag="global")

                for client in self.selected_clients:
                    client.train(task=task)
                
                self.receive_models()
                self.train_global_generator() 
                self.aggregate_parameters()
                
                self.Budget.append(time.time() - s_t)
                print(f"Round {i} Time: {self.Budget[-1]:.2f}s")

            # --- Post-Task Processing ---
            self.receive_models()
            self.train_global_generator()
            self.train_global_classifier()
            
            print(f"\n>>> Task {task} finished. Evaluating Forgetting Rate...") 
            self.eval_task(task=task, glob_iter=task, flag="global")
            self.send_models()

    def train_global_generator(self):
        print(f"[Server] Start training Generator (with BN Reg)")
        
        # 1. Prepare Labels
        processed_client_labels = {}
        all_labels_set = set()

        for client_id, info in self.client_info_dict.items():
            raw_labels = info["label"]
            # Convert various iterables to numpy array
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
            
            # Freeze teacher parameters
            info["model"].eval()
            for param in info["model"].parameters(): param.requires_grad = False

        labels_list = list(all_labels_set)
        if len(labels_list) == 0: return

        # [FIX] Unfreeze Generator
        self.global_generator.train()
        for param in self.global_generator.parameters(): param.requires_grad = True

        criterion_ce = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss() 
        alpha_anchor = 10.0
        beta_bn = 0.5 

        for step in range(self.g_steps):
            self.optimizer_g.zero_grad()
            
            # A. Generate
            selected_labels = np.random.choice(labels_list, self.batch_size_gen)
            labels_tensor = torch.tensor(selected_labels, dtype=torch.long).to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)
            gen_imgs = self.global_generator(z, labels_tensor)

            # B. Teacher Losses
            total_ce_loss = torch.tensor(0.0, device=self.device)
            total_bn_loss = torch.tensor(0.0, device=self.device)
            valid_teachers = 0
            
            for client_id, info in self.client_info_dict.items():
                teacher_model = info["model"]
                teacher_labels = processed_client_labels[client_id]
                
                mask = np.isin(selected_labels, teacher_labels)
                if mask.sum() > 0:
                    valid_teachers += 1
                    mask_tensor = torch.tensor(mask, device=self.device)
                    relevant_imgs = gen_imgs[mask_tensor]
                    relevant_labels = labels_tensor[mask_tensor]
                    
                    preds = teacher_model(relevant_imgs)
                    total_ce_loss += criterion_ce(preds, relevant_labels)
                    total_bn_loss += get_bn_loss(teacher_model, relevant_imgs)

            loss_g = total_ce_loss + (beta_bn * total_bn_loss)
            
            # C. Anchor Loss
            if self.prev_generator is not None:
                self.prev_generator.eval()
                for param in self.prev_generator.parameters(): param.requires_grad = False
                with torch.no_grad():
                    anchor_imgs = self.prev_generator(z, labels_tensor)
                loss_g += alpha_anchor * mse_loss(gen_imgs, anchor_imgs)

            if valid_teachers > 0:
                loss_g.backward()
                self.optimizer_g.step()
                self.scheduler_g.step()
        
        # Update Anchor
        self.prev_generator = copy.deepcopy(self.global_generator)

    def train_global_classifier(self, steps=100):
        print(f"[Server] Start training Global Classifier")
        all_labels = set()
        for v in self.client_info_dict.values():
            raw_labels = v["label"]
            if isinstance(raw_labels, torch.Tensor): raw_labels = raw_labels.cpu().tolist()
            elif isinstance(raw_labels, np.ndarray): raw_labels = raw_labels.tolist()
            all_labels.update(list(raw_labels))
        labels_list = list(all_labels)

        if len(labels_list) == 0: return

        self.global_generator.eval()
        for param in self.global_generator.parameters(): param.requires_grad = False
        
        # [FIX] Unfreeze Global Model
        self.global_model.train()
        for param in self.global_model.parameters(): param.requires_grad = True
        
        for step in range(steps):
            self.optimizer_c.zero_grad()
            
            # 1. Generate (Detach)
            selected_labels = np.random.choice(labels_list, self.batch_size_gen)
            labels_tensor = torch.tensor(selected_labels).long().to(self.device)
            z = torch.randn(self.batch_size_gen, self.nz).to(self.device)
            with torch.no_grad():
                gen_imgs = self.global_generator(z, labels_tensor).detach()

            # 2. Student
            student_logits = self.global_model(gen_imgs)

            # 3. Teachers
            teacher_logits_sum = torch.zeros_like(student_logits)
            teacher_counts = torch.zeros(self.batch_size_gen, 1, device=self.device)
            
            for client_id, info in self.client_info_dict.items():
                teacher_model = info["model"]
                mask = np.isin(selected_labels, info["label"])
                
                if mask.sum() > 0:
                    mask_tensor = torch.tensor(mask, device=self.device).unsqueeze(1)
                    teacher_model.eval()
                    with torch.no_grad():
                        logits = teacher_model(gen_imgs)
                    teacher_logits_sum += logits * mask_tensor
                    teacher_counts += mask_tensor
            
            teacher_counts[teacher_counts == 0] = 1.0
            teacher_avg_logits = teacher_logits_sum / teacher_counts
            
            loss = KD_loss(student_logits, teacher_avg_logits, self.T)
            loss.backward()
            self.optimizer_c.step()
            self.scheduler_c.step()

    def send_models(self):
        for client in self.clients:
            client.set_parameters(self.global_model)
            client.set_generator_parameters(self.global_generator)

    def receive_models(self):
        print(f"[Server-side] Receive models and lists of labels from clients")
        self.client_info_dict = {}
        self.uploaded_models = []
        for client in self.clients:
            # [FIX] Use classes_so_far for reliability
            clean_labels = list(set(client.classes_so_far))
            self.client_info_dict[client.id] = {
                "model": client.model,
                "label": clean_labels
            }
            self.uploaded_models.append(client.model)
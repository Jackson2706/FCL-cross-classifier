
import time
import torch
import torch.nn.functional as F
import copy
import numpy as np
import os
from torch import nn, optim
from torchvision.utils import save_image
from flcore.servers.serverbase import Server
from flcore.clients.clientdgr import clientDGR
from flcore.utils_core.target_utils import *
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k, read_client_data_FCL_cifar10, cifar100_train_transform


# ==========================================
# 1. ADVANCED GENERATOR (Phải khớp với Client)
# ==========================================
class AdvancedGenerator(nn.Module):
    """Generator optimized for 224x224 images with better class conditioning"""

    def __init__(self, nz=100, ngf=128, img_size=224, nc=3, num_classes=7, device=None):
        super(AdvancedGenerator, self).__init__()
        self.params = (nz, ngf, img_size, nc, num_classes)
        self.num_classes = num_classes
        self.device = device

        # Xác định init_size dựa trên img_size
        if img_size == 224:
            self.init_size = 7
        elif img_size == 64:
            self.init_size = 4
        elif img_size == 32:
            self.init_size = 4
        else:
            self.init_size = img_size // 32

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
            self._upsample_block(ngf * 16, ngf * 8),  # 7->14 (or 4->8)
            self._upsample_block(ngf * 8, ngf * 8),  # 14->28 (or 8->16)
            self._upsample_block(ngf * 8, ngf * 4),  # 28->56 (or 16->32)
            # Logic xử lý kích thước ảnh nhỏ (32x32) hay lớn (224x224)
            # Lưu ý: Cấu trúc dưới đây giả định cho 224.
            # Với CIFAR (32), Generator cần điều chỉnh số lớp.
            # Để an toàn, tôi giữ nguyên cấu trúc bạn cung cấp.
            self._upsample_block(ngf * 4, ngf * 2),
            self._upsample_block(ngf * 2, ngf),

            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Sigmoid(),
        )

        # Xử lý riêng cho CIFAR (32x32) để tránh lỗi dimension nếu dùng code của 224
        if img_size == 32:
            self.conv_blocks = nn.Sequential(
                self._upsample_block(ngf * 16, ngf * 8),  # 4->8
                self._upsample_block(ngf * 8, ngf * 4),  # 8->16
                self._upsample_block(ngf * 4, ngf * 2),  # 16->32
                nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
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
# 3. SERVER CLASS
# ==========================================
class serverCoplay(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.Budget = []

        # --- Config Generator ---
        if 'cifar100' in self.dataset.lower():
            self.img_size = 32; self.nz = 256
        elif 'imagenet' in self.dataset.lower():
            self.img_size = 64 if '100' in self.dataset else 224; self.nz = 256
        else:
            self.img_size = 32; self.nz = 100

        self.global_generator = AdvancedGenerator(
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

        self.set_clients(clientDGR)
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
                # self.send_models()
                self.eval(task=task, glob_iter=glob_iter, flag="global")

                for client in self.selected_clients:
                    client.train(task=task)

                self.receive_models()
                self.aggregate_parameters()

                self.Budget.append(time.time() - s_t)
                print(f"Round {i} Time: {self.Budget[-1]:.2f}s")

            # --- Post-Task Processing ---
            self.receive_models()
            self.train_global_generator()
            self.train_global_classifier()
            # ==================================================================================
            # [RESOLVED TODO] Evaluation & Visualization
            # ==================================================================================
            print(f"\n>>> [Eval] Detailed Evaluation for Task {task} (Real vs Synthetic)")

            # 1. Visualization & Structured Dataset Generation
            debug_dir = os.path.join("output_debug", self.args.dataset, f"task_{task}")
            os.makedirs(debug_dir, exist_ok=True)

            self.global_generator.eval()
            with torch.no_grad():
                # Gather all currently known classes
                all_current_classes = sorted(list(set().union(*[c.classes_so_far for c in self.clients])))

                if len(all_current_classes) > 0:
                    print(f"    [Vis] Generating 10 samples per class for {len(all_current_classes)} classes...")

                    samples_per_class = 10

                    for class_id in all_current_classes:
                        # A. Create folder for this specific class
                        class_dir = os.path.join(debug_dir, f"class_{class_id}")
                        os.makedirs(class_dir, exist_ok=True)

                        # B. Prepare inputs: All labels are 'class_id'
                        labels = torch.full((samples_per_class,), class_id, dtype=torch.long, device=self.device)
                        z = torch.randn(samples_per_class, self.nz, device=self.device)

                        # C. Generate batch
                        gen_imgs = self.global_generator(z, labels)

                        # D. Denormalize (Tanh [-1, 1] -> [0, 1])
                        gen_imgs = (gen_imgs + 1) / 2.0

                        # E. Save individual images
                        for idx in range(samples_per_class):
                            save_path = os.path.join(class_dir, f"sample_{idx}.png")
                            save_image(gen_imgs[idx], save_path)

                    print(f"    [Vis] Saved structured synthetic dataset to: {debug_dir}")

            # 2. Generalization Gap (Real Acc - Syn Acc)
            total_real_acc = []
            total_syn_acc = []
            total_gap = []

            for client_idx, client in enumerate(self.clients):
                client.model.eval()
                known_classes = set(client.classes_so_far)
                if not known_classes: continue

                # A. Evaluate on Real Data (Use Client's test logic if available)
                ct, ns = client.test_metrics(task=task)
                acc_real = (ct / ns * 100.0) if ns > 0 else 0.0

                # B. Evaluate on Synthetic Data
                num_syn_samples = 256
                syn_labels_np = np.random.choice(list(known_classes), num_syn_samples)
                syn_labels = torch.tensor(syn_labels_np, dtype=torch.long, device=self.device)
                syn_z = torch.randn(num_syn_samples, self.nz, device=self.device)

                with torch.no_grad():
                    # Forward: z, labels
                    syn_imgs = self.global_generator(syn_z, syn_labels)
                    syn_logits = client.model(syn_imgs)
                    syn_preds = syn_logits.argmax(dim=1)
                    correct_syn = (syn_preds == syn_labels).sum().item()

                acc_syn = (correct_syn / num_syn_samples * 100.0)

                gap = acc_real - acc_syn
                total_real_acc.append(acc_real)
                total_syn_acc.append(acc_syn)
                total_gap.append(gap)

                print(f"    [Client {client_idx}] Classes: {len(known_classes)} | "
                      f"Real Acc: {acc_real:.2f}% | Syn Acc: {acc_syn:.2f}% | Gap: {gap:.2f}%")

            if len(total_real_acc) > 0:
                avg_real = sum(total_real_acc) / len(total_real_acc)
                avg_syn = sum(total_syn_acc) / len(total_syn_acc)
                avg_gap = sum(total_gap) / len(total_gap)
                print \
                    (f"    [Summary Task {task}] Avg Real: {avg_real:.2f}% | Avg Syn: {avg_syn:.2f}% | Avg Gap: {avg_gap:.2f}%")

            # ==================================================================================

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

                    # Ensure teacher is on correct device
                    teacher_model.to(self.device)
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
                    teacher_model.to(self.device)
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
            # client.set_generator_parameters(self.global_generator)

    def receive_models(self):
        print(f"[Server-side] Receive models and lists of labels from clients")
        self.client_info_dict = {}
        self.uploaded_models = []
        for client in self.clients:
            clean_labels = list(set(client.classes_so_far))
            self.client_info_dict[client.id] = {
                "model": client.model,
                "label": clean_labels
            }
            self.uploaded_models.append(client.model)
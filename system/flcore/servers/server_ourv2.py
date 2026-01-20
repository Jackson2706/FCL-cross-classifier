import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from flcore.clients.client_ourv2 import clientOursV2
from flcore.servers.serverbase import Server
from torch import nn, optim
from torch.nn.utils import spectral_norm
from torchvision.utils import save_image
from utils.data_utils import (read_client_data_FCL_cifar10,
                              read_client_data_FCL_cifar100,
                              read_client_data_FCL_imagenet1k)


class BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_feature = None 

    def hook_fn(self, module, input, output):
        # We capture the input to the BatchNorm layer
        self.r_feature = input[0]

    def remove(self):
        self.hook.remove()
        
# ==========================================
# 1. ADVANCED GENERATOR & UTILS
# ==========================================
class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

def denormalize(tensor, mean, std):
    """Reverses the normalization for visualization"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

class AdvancedGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, num_classes=10, device=None):
        super(AdvancedGenerator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.init_size = img_size // 8
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.l1 = nn.Sequential(
            nn.Linear(nz + num_classes, ngf * 8 * self.init_size ** 2),
            nn.BatchNorm1d(ngf * 8 * self.init_size ** 2),
            nn.ReLU(True)
        )

        self.conv_blocks = nn.Sequential(
            self._upsample_block(ngf*8, ngf*4), 
            self._upsample_block(ngf*4, ngf*2),  
            self._upsample_block(ngf*2, ngf),
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Sigmoid() 
        )
        
        # Consistent normalization for CIFAR
        self.stats = {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]}
        self.norm = NormalizeLayer(self.stats['mean'], self.stats['std'])

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, z, labels):
        gen_input = torch.cat([z, self.label_emb(labels)], dim=1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return self.norm(img)

class Critic(nn.Module):
    def __init__(self, nc=3, ndf=64, num_classes=10, img_size=32):
        super(Critic, self).__init__()
        self.img_size = img_size
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc + 1, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, 1, 4, 2, 1)),
        )

    def forward(self, img, labels):
        label_embed = self.label_embedding(labels).view(-1, 1, self.img_size, self.img_size)
        d_in = torch.cat((img, label_embed), dim=1)
        return self.main(d_in).view(-1, 1)

# ==========================================
# 2. SERVER CLASS
# ==========================================
class OursV2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.img_size = 32 if 'cifar' in self.dataset.lower() else 224
        self.nz = 256 if 'cifar100' in self.dataset.lower() else 100

        self.global_generator = AdvancedGenerator(nz=self.nz, img_size=self.img_size, num_classes=args.num_classes).to(self.device)
        self.critic = Critic(num_classes=args.num_classes, img_size=self.img_size).to(self.device)
        self.optimizer_g = optim.Adam(self.global_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_cr = optim.Adam(self.critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_c = optim.Adam(self.global_model.parameters(), lr=getattr(args, 'c_lr', 0.001))
        
        self.prev_generator = None
        self.set_clients(clientOursV2)

    def train(self):
        for task in range(self.args.num_tasks):
            print(f"\n--- Task {task} ---")
            if task > 0:
                self._update_client_data(task)

            for i in range(self.global_rounds):
                self.selected_clients = self.select_clients()
                for client in self.selected_clients:
                    client.train(task=task)
                self.receive_models()
                self.aggregate_parameters()
                self.send_models()
                self.eval(task=task, glob_iter=i + task*self.global_rounds, flag="global")

            self.train_global_generator()
            self.train_global_classifier()
            self.visualize_synthetic_data(task)
            self.eval_task(task=task, glob_iter=task, flag="global")
            self.send_models()

    def _update_client_data(self, task):
        for i, client in enumerate(self.clients):
            read_func = read_client_data_FCL_cifar100 if 'cifar100' in self.args.dataset.lower() else read_client_data_FCL_cifar10
            train_data, label_info = read_func(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
            client.next_task(train_data, label_info)

    def train_global_generator(self):
        self.global_generator.train()
        self.critic.train()
        criterion_ce = nn.CrossEntropyLoss()
        MIN_BN_SAMPLES = 16 # Stabilize BN loss

        for _ in range(getattr(self.args, 'g_steps', 200)):
            z = torch.randn(64, self.nz).to(self.device)
            labels = torch.randint(0, self.args.num_classes, (64,)).to(self.device)

            # 1. Update Critic
            self.optimizer_cr.zero_grad()
            gen_imgs = self.global_generator(z, labels)
            d_loss = -torch.mean(self.critic(gen_imgs.detach(), labels))
            d_loss.backward()
            self.optimizer_cr.step()

            # 2. Update Generator
            self.optimizer_g.zero_grad()
            loss_adv = -torch.mean(self.critic(gen_imgs, labels))
            
            total_ce, total_bn, valid_t = 0, 0, 0
            for _, info in self.client_info_dict.items():
                mask = np.isin(labels.cpu().numpy(), info["label"])
                if mask.sum() > 0:
                    valid_t += 1
                    m_idx = torch.tensor(mask, device=self.device)
                    preds = info["model"].eval().to(self.device)(gen_imgs[m_idx])
                    total_ce += criterion_ce(preds, labels[m_idx])
                    if mask.sum() >= MIN_BN_SAMPLES: # Check threshold
                        total_bn += self.get_bn_loss(info["model"], gen_imgs[m_idx])

            loss_g = loss_adv + (total_ce / max(1, valid_t)) + 0.1 * (total_bn / max(1, valid_t))
            loss_g.backward()
            self.optimizer_g.step()

    def get_bn_loss(self, teacher_model, gen_imgs):
        """Calculates BN statistics distance between teacher and fake data."""
        bn_hooks = []
        # Identify all 2D BatchNorm layers in the teacher model
        bn_layers = [m for m in teacher_model.modules() if isinstance(m, nn.BatchNorm2d)]

        for module in bn_layers:
            bn_hooks.append(BNFeatureHook(module))

        # Forward pass to trigger hooks and capture internal features
        teacher_model(gen_imgs)

        loss_bn = 0.0
        for hook, layer in zip(bn_hooks, bn_layers):
            # Stats from the teacher model (Real data)
            real_mean = layer.running_mean
            real_var = layer.running_var
            
            # Stats from the current batch of generated images
            gen_feat = hook.r_feature
            gen_mean = torch.mean(gen_feat, dim=[0, 2, 3])
            gen_var = torch.var(gen_feat, dim=[0, 2, 3], unbiased=False)
            
            # Calculate L2 norm difference
            loss_bn += torch.norm(gen_mean - real_mean, 2) + torch.norm(gen_var - real_var, 2)

        # Clean up hooks to prevent memory leaks
        for hook in bn_hooks: 
            hook.remove()
        
        return loss_bn

    def train_global_classifier(self):
        self.global_model.train()
        self.global_generator.eval()
        for _ in range(100):
            z = torch.randn(64, self.nz).to(self.device)
            labels = torch.randint(0, self.args.num_classes, (64,)).to(self.device)
            with torch.no_grad():
                imgs = self.global_generator(z, labels)
            
            self.optimizer_c.zero_grad()
            logits = self.global_model(imgs)
            # Use instance KD loss
            loss = self.KD_loss(logits, labels, T=2.0) 
            loss.backward()
            self.optimizer_c.step()

    def KD_loss(self, student_logits, labels, T=2.0):
        # Target is hard labels here for simplicity, or soft-logits from teachers
        return F.cross_entropy(student_logits / T, labels)

    def visualize_synthetic_data(self, task):
        debug_dir = os.path.join("output_debug", self.args.dataset, f"task_{task}")
        os.makedirs(debug_dir, exist_ok=True)
        
        self.global_generator.eval()
        
        # 1. Collect all classes seen across all clients so far
        all_seen_classes = set()
        for client in self.clients:
            all_seen_classes.update(client.classes_so_far)
        
        # Convert to sorted list for consistent grid visualization
        all_seen_classes = sorted(list(all_seen_classes))
        
        if not all_seen_classes:
            print("No seen classes found to visualize.")
            return

        print(f"[Vis] Generating samples for {len(all_seen_classes)} seen classes: {all_seen_classes}")

        with torch.no_grad():
            # 2. Prepare inputs for all seen classes
            # Generating 1 sample per class for a clean overview grid
            labels = torch.tensor(all_seen_classes, dtype=torch.long).to(self.device)
            z = torch.randn(len(labels), self.nz).to(self.device)
            
            # 3. Generate and Denormalize
            imgs = self.global_generator(z, labels)
            
            # Ensure denormalize uses the specific stats defined in your AdvancedGenerator
            stats = self.global_generator.stats
            imgs = denormalize(imgs.cpu(), stats['mean'], stats['std'])
            
            # 4. Save as a grid (nrow can be adjusted based on number of tasks/classes)
            save_path = os.path.join(debug_dir, f"all_seen_classes_task_{task}.png")
            save_image(imgs, save_path, nrow=int(np.sqrt(len(all_seen_classes))) + 1)
            print(f"[Vis] Saved all-class grid to {save_path}")

    def receive_models(self):
        self.client_info_dict = {c.id: {"model": copy.deepcopy(c.model), "label": list(c.classes_so_far)} for c in self.selected_clients}
        super().receive_models()

    def send_models(self):
        for client in self.clients:
            client.set_parameters(self.global_model)
            client.set_generator_parameters(self.global_generator)
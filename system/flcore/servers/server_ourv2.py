import copy
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from flcore.clients.client_ourv2 import clientOursV2
from flcore.scheduler.consolidation import ConsolidationManager
from flcore.servers.serverbase import Server
from flcore.scheduler.task_scheduler import TaskScheduler
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
# 1. Type A: Lowest Complexity (MLP)
# Perfect for strict resource-constrained testing.
# ==========================================
class MLPGenerator(nn.Module):
    def __init__(self, nz=100, img_size=32, nc=3, num_classes=10, device=None):
        super(MLPGenerator, self).__init__()
        self.nc = nc
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.net = nn.Sequential(
            nn.Linear(nz + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, nc * img_size * img_size),
            nn.Sigmoid()
        )
        self.stats = {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]}
        self.norm = NormalizeLayer(self.stats['mean'], self.stats['std'])

    def forward(self, z, labels):
        gen_input = torch.cat([z, self.label_emb(labels)], dim=1)
        out = self.net(gen_input)
        img = out.view(-1, self.nc, self.img_size, self.img_size)
        return self.norm(img)

# ==========================================
# 3. Type C: Medium-High Complexity (Advanced CNN)
# Your current architecture. (Included here just 
# to show where it sits in the spectrum).
# ==========================================
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

# ==========================================
# 2. Type B: Low-Medium Complexity (Light CNN)
# A stripped-down version of your current CNN.
# ==========================================
class LightCNNGenerator(AdvancedGenerator):
    def __init__(self, nz=100, img_size=32, nc=3, num_classes=10, device=None):
        # Inherits from your AdvancedGenerator but slashes the feature maps (ngf) from 64 to 16
        super().__init__(nz=nz, ngf=16, img_size=img_size, nc=nc, num_classes=num_classes, device=device)
        
# ==========================================
# 4. Type D: High Complexity (ResNet-Based)
# Strong baseline for high-fidelity generation.
# ==========================================
class UpResBlock(nn.Module):
    """Residual upsampling block for the ResNet Generator"""
    def __init__(self, in_channels, out_channels):
        super(UpResBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        
        # Shortcut connection for the residual
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(self.up(x))))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNetGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, num_classes=10, device=None):
        super(ResNetGenerator, self).__init__()
        self.init_size = img_size // 8
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.l1 = nn.Sequential(
            nn.Linear(nz + num_classes, ngf * 8 * self.init_size ** 2),
            nn.BatchNorm1d(ngf * 8 * self.init_size ** 2),
            nn.ReLU(True)
        )

        self.res_blocks = nn.Sequential(
            UpResBlock(ngf * 8, ngf * 4),
            UpResBlock(ngf * 4, ngf * 2),
            UpResBlock(ngf * 2, ngf),
            nn.Conv2d(ngf, nc, 3, 1, 1),
            nn.Sigmoid()
        )
        
        self.stats = {'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]}
        self.norm = NormalizeLayer(self.stats['mean'], self.stats['std'])

    def forward(self, z, labels):
        gen_input = torch.cat([z, self.label_emb(labels)], dim=1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        img = self.res_blocks(out)
        return self.norm(img)

# ==========================================
# 2. SERVER CLASS
# ==========================================
class OursV2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.img_size = 32 if 'cifar' in self.dataset.lower() else 224
        self.nz = 256 if 'cifar100' in self.dataset.lower() else 100
        self.generated_samples_per_class = args.generated_samples_per_class if hasattr(args, 'generated_samples_per_class') else 100
        
        # New argument for the robustness filter
        self.filter_threshold = getattr(args, 'filter_threshold', 1.5)

        gen_type = getattr(args, 'gen_type', 'advanced').lower()
        print(f"\n[Server] Initializing generator architecture: {gen_type.upper()}")
        
        generator_kwargs = {
            'nz': self.nz, 
            'img_size': self.img_size, 
            'num_classes': args.num_classes,
            'device': self.device
        }

        if gen_type == 'mlp':
            self.global_generator = MLPGenerator(**generator_kwargs).to(self.device)
        elif gen_type == 'light_cnn':
            self.global_generator = LightCNNGenerator(**generator_kwargs).to(self.device)
        elif gen_type == 'advanced':
            self.global_generator = AdvancedGenerator(**generator_kwargs).to(self.device)
        elif gen_type == 'resnet':
            self.global_generator = ResNetGenerator(**generator_kwargs).to(self.device)
        else:
            raise ValueError(f"Unknown gen_type: {gen_type}. Choose from: mlp, light_cnn, advanced, resnet.")

        self.critic = Critic(num_classes=args.num_classes, img_size=self.img_size).to(self.device)
        self.optimizer_g = optim.Adam(self.global_generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_cr = optim.Adam(self.critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_c = optim.Adam(self.global_model.parameters(), lr=getattr(args, 'c_lr', 0.001))
        
        self.prev_generator = None
        self.set_clients(clientOursV2)

        async_mode = getattr(args, 'async_mode', False)
        self.async_mode = bool(async_mode)
        scheduler_mode = (
            getattr(args, 'client_task_speed_distribution', 'fixed_groups')
            if async_mode else 'synchronous'
        )
        scheduler_kwargs = dict(
            num_clients=self.num_clients,
            num_tasks=self.num_tasks,
            rounds_per_task=self.global_rounds,
            mode=scheduler_mode,
            max_task_lag=getattr(args, 'max_task_lag', 0),
            client_dropout_rate=getattr(args, 'client_dropout_rate', 0.0),
            permanent_dropout_rate=getattr(args, 'permanent_dropout_rate', 0.0),
            partial_participation_rate=getattr(args, 'partial_participation_rate', 1.0),
            seed=getattr(args, 'task_schedule_seed', 0),
            num_speed_groups=getattr(args, 'num_speed_groups', 2),
            speed_interval=getattr(args, 'speed_interval', [0.5, 1.0]),
            custom_schedule_path=getattr(args, 'custom_schedule_path', None),
            allow_task_jumps=getattr(args, 'allow_task_jumps', False),
        )
        self.task_scheduler = TaskScheduler(**scheduler_kwargs)
        self._client_loaded_task = {client.id: 0 for client in self.clients}
        self._evaluated_client_stages = set()
        if self.async_mode:
            total_rounds = self.num_tasks * self.global_rounds
            schedule_forecast = TaskScheduler(**scheduler_kwargs)
            self._async_schedule = {
                global_round: schedule_forecast.state_for_round(global_round)
                for global_round in range(total_rounds)
            }
            permanent_dropped = set(
                schedule_forecast.materialized_schedule()["permanent_dropped_clients"]
            )
            eligible_by_boundary = {
                boundary_k: {
                    client.id for client in self.clients
                    if client.id not in permanent_dropped
                    and any(
                        states[client.id].task_id == boundary_k
                        for states in self._async_schedule.values()
                    )
                }
                for boundary_k in range(self.num_tasks)
            }
            self.consolidation_manager = ConsolidationManager(
                eligible_by_boundary,
                trigger=getattr(args, 'consolidation_trigger', 'watermark'),
                timeout_rounds=getattr(args, 'consolidation_timeout_rounds', None),
                quorum=getattr(args, 'consolidation_quorum', 1.0),
            )
            self._async_last_task_upload = {}

    def filter_anomalous_clients(self):
        """
        Filters out corrupted or low-quality client classifiers using L2 weight divergence
        from the current global model. Modifies self.client_info_dict in place.
        """
        if not self.client_info_dict or len(self.client_info_dict) == 0:
            return

        global_weights = torch.cat([p.view(-1) for p in self.global_model.parameters()])
        divergences = {}
        
        # Calculate L2 distance for each client in the dict
        for client_id, info in self.client_info_dict.items():
            client_model = info["model"]
            client_weights = torch.cat([p.view(-1) for p in client_model.parameters()])
            l2_dist = torch.norm(client_weights - global_weights, p=2).item()
            divergences[client_id] = l2_dist
            
        # Calculate dynamic threshold based on the median divergence
        div_values = list(divergences.values())
        if len(div_values) == 0:
            return
            
        median_div = torch.median(torch.tensor(div_values)).item()
        threshold = median_div * self.filter_threshold
        
        # Filter clients
        dropped_ids = []
        safe_dict = {}
        for client_id, div in divergences.items():
            if div <= threshold:
                safe_dict[client_id] = self.client_info_dict[client_id]
            else:
                dropped_ids.append(client_id)
                
        if len(dropped_ids) > 0:
            print(f"[Server Filter] Dropped {len(dropped_ids)} anomalous clients based on L2 divergence: {dropped_ids}")
            
        # Update the dictionary to only contain safe clients for generator training
        self.client_info_dict = safe_dict

    def train(self):
        if self.async_mode:
            return self._train_async()

        for task in self.task_scheduler.task_sequence():
            task_states = self.task_scheduler.state_for_round(task * max(1, self.global_rounds))
            print(f"\n--- Task {task} ---")
            if task > 0:
                self._update_client_data(task_states)

            for i in range(self.global_rounds):
                global_round = i + task * self.global_rounds
                round_states = self.task_scheduler.state_for_round(global_round)
                self.selected_clients = self.select_clients()
                
                # SIMULATE ATTACK/NOISE: Inject massive noise into the first 2 selected clients
                # Remove or comment out this block when running normal, non-poisoned experiments!
                if getattr(self.args, 'simulate_bad_clients', False):
                    bad_client_count = int(len(self.selected_clients) * 0.20) # 20% bad clients
                    for j in range(bad_client_count):
                        print(f"[Simulate Attack] Injecting severe noise into client ID: {self.selected_clients[j].id}")
                        for param in self.selected_clients[j].model.parameters():
                            param.data += torch.randn_like(param.data) * 5.0 # Massive noise
                
                for client in self.selected_clients:
                    client_state = round_states[client.id]
                    if client_state.active and not client_state.dropped:
                        client.train(task=client_state.task_id)
                
                self.receive_models()
                self.aggregate_parameters()
                self.send_models()
                self.eval(task=task, glob_iter=global_round, flag="global")

            # --- NEW: Filter anomalous client classifiers BEFORE Coplay ---
            if getattr(self.args, 'use_filter', True):
                self.filter_anomalous_clients()
            # --------------------------------------------------------------

            with self.server_compute_timer.measure(task):
                self.train_global_generator()
                self.train_global_classifier()
            self.visualize_synthetic_data(task)
            self.eval_task(task=task, glob_iter=task, flag="global")
            self.send_models()
            self._write_metrics_summary()

    def _train_async(self):
        """Run asynchronous client clocks with completion-watermark consolidation."""

        total_rounds = self.num_tasks * self.global_rounds
        previous_states = None
        for global_round in range(total_rounds):
            round_states = self.task_scheduler.state_for_round(global_round)
            pending_completions = []
            for client in self.clients:
                state = round_states[client.id]
                self.consolidation_manager.mark_reached(
                    state.task_id, client.id, global_round
                )
                if previous_states is None:
                    continue
                previous_task = previous_states[client.id].task_id
                if state.task_id <= previous_task:
                    continue
                upload = self._async_last_task_upload.get(client.id)
                if upload is not None and upload["task_id"] == previous_task:
                    pending_completions.append(upload)

            self._update_client_data(round_states)
            self.selected_clients = [
                client for client in self.clients
                if round_states[client.id].active and not round_states[client.id].dropped
            ]

            if getattr(self.args, 'simulate_bad_clients', False):
                bad_client_count = int(len(self.selected_clients) * 0.20)
                for client in self.selected_clients[:bad_client_count]:
                    print(f"[Simulate Attack] Injecting severe noise into client ID: {client.id}")
                    for param in client.model.parameters():
                        param.data += torch.randn_like(param.data) * 5.0

            for client in self.selected_clients:
                client.train(task=round_states[client.id].task_id)
                self._cache_async_task_upload(client, round_states[client.id].task_id)

            self._receive_async_models()
            if self.uploaded_models:
                self.aggregate_parameters()
            else:
                print(f"[Async] Round {global_round}: no fresh uploads; global model unchanged.")
            self.send_models()
            global_stage = min(global_round // max(1, self.global_rounds), self.num_tasks - 1)
            self.eval(task=global_stage, glob_iter=global_round, flag="global")
            self._record_new_client_stage_metrics(round_states)

            for upload in pending_completions:
                self._publish_async_completion(upload, global_round)
            self._consolidate_ready_boundaries(global_round)
            previous_states = round_states

        # A client's last stage has no later clock transition to publish it.
        final_round = max(0, total_rounds - 1)
        for client in self.clients:
            upload = self._async_last_task_upload.get(client.id)
            if upload is not None:
                self._publish_async_completion(upload, final_round)
        self._consolidate_ready_boundaries(final_round)

        self._write_metrics_summary()

    def _cache_async_task_upload(self, client, task_id):
        """Retain the local upload before the next global broadcast replaces it."""

        self._async_last_task_upload[client.id] = {
            "client_id": client.id,
            "task_id": int(task_id),
            "model": copy.deepcopy(client.model),
            "class_labels": list(client.task_dict[int(task_id)]),
            "sample_count": len(client.train_data),
        }

    def _publish_async_completion(self, upload, global_round):
        late_arrival_count = len(self.consolidation_manager.late_arrivals)
        recorded = self.consolidation_manager.record_completion(
            boundary_k=upload["task_id"],
            client_id=upload["client_id"],
            model=upload["model"],
            class_labels=upload["class_labels"],
            sample_count=upload["sample_count"],
            global_round=global_round,
        )
        if (
            not recorded
            and len(self.consolidation_manager.late_arrivals) > late_arrival_count
        ):
            print(
                f"[Async][Consolidation] Late completion for boundary "
                f"{upload['task_id']} from client {upload['client_id']}; not re-running."
            )

    def _consolidate_ready_boundaries(self, global_round):
        for event in self.consolidation_manager.pop_ready(global_round):
            print(
                f"[Async][Consolidation] Boundary {event.boundary_k} READY via "
                f"{event.trigger}; clients={list(event.participating_client_ids)}, "
                f"missing={list(event.missing_client_ids)}."
            )
            previous_client_info = self.client_info_dict
            self.client_info_dict = {
                snapshot.client_id: {
                    "model": copy.deepcopy(snapshot.model),
                    "label": list(snapshot.class_labels),
                    "sample_count": snapshot.sample_count,
                    "bn_statistics": copy.deepcopy(snapshot.bn_statistics),
                }
                for snapshot in event.teacher_snapshots
            }
            try:
                if getattr(self.args, 'use_filter', True):
                    self.filter_anomalous_clients()
                with self.server_compute_timer.measure(event.boundary_k):
                    self.train_global_generator(
                        seen_classes=event.globally_consolidated_classes
                    )
                    self.train_global_classifier(
                        seen_classes=event.globally_consolidated_classes
                    )
            finally:
                self.client_info_dict = previous_client_info
            self.eval_task(
                task=event.boundary_k,
                glob_iter=global_round,
                flag="global",
            )
            self.send_models()
            self._write_metrics_summary()

    def _receive_async_models(self):
        """Accept only fresh uploads and renormalize sample weights over them."""

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.client_info_dict = {}
        total_samples = sum(len(client.train_data) for client in self.selected_clients)
        if total_samples == 0:
            return
        for client in self.selected_clients:
            sample_count = len(client.train_data)
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(sample_count / total_samples)
            self.uploaded_models.append(client.model)
            self.client_info_dict[client.id] = {
                "model": copy.deepcopy(client.model),
                "label": list(client.classes_so_far),
            }
        self.communication_accountant.record_uplink(
            self.uploaded_models[0], len(self.uploaded_models)
        )

    def _record_new_client_stage_metrics(self, round_states):
        """Evaluate tasks 0..s once, just after each client first reaches stage s."""

        for client in self.clients:
            stage = round_states[client.id].task_id
            key = (client.id, stage)
            if key in self._evaluated_client_stages:
                continue
            accuracies, sample_counts = [], []
            for task_id in range(stage + 1):
                correct, count = client.test_metrics(task=task_id)
                accuracies.append(correct / count if count else 0.0)
                sample_counts.append(count)
            self.task_scheduler.record_client_stage_accuracy(
                client.id, stage, accuracies, sample_counts
            )
            self._evaluated_client_stages.add(key)

    def _update_client_data(self, client_states):
        for i, client in enumerate(self.clients):
            state = client_states[client.id]
            loaded_task = self._client_loaded_task.get(client.id, client.current_task)
            for next_task in range(loaded_task + 1, state.task_id + 1):
                if 'cifar100' in self.args.dataset.lower():
                    read_func = read_client_data_FCL_cifar100
                elif 'cifar10' in self.args.dataset.lower():
                    read_func = read_client_data_FCL_cifar10
                elif 'imagenet1k' in self.args.dataset.lower():
                    read_func = read_client_data_FCL_imagenet1k
                else:
                    raise NotImplementedError(f"Async task loading is unsupported for {self.args.dataset}")
                train_data, label_info = read_func(
                    i, task=next_task, classes_per_task=self.args.cpt, count_labels=True
                )
                client.next_task(train_data, label_info)
                self._client_loaded_task[client.id] = next_task

    def train_global_generator(self, seen_classes=None):
        self.global_generator.train()
        self.critic.train()
        criterion_ce = nn.CrossEntropyLoss()
        MIN_BN_SAMPLES = 16 # Stabilize BN loss

        # Get only seen classes
        seen_classes = self.get_seen_classes() if seen_classes is None else list(seen_classes)
        if not seen_classes:
            print("No seen classes yet. Skipping generator training.")
            return
        seen_classes_tensor = torch.tensor(seen_classes, dtype=torch.long, device=self.device)

        for _ in range(getattr(self.args, 'g_steps', 200)):
            z = torch.randn(64, self.nz).to(self.device)
            
            # Sample only from seen classes
            idx = torch.randint(0, len(seen_classes), (64,), device=self.device)
            labels = seen_classes_tensor[idx]

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
            
            # Loop runs over the filtered safe dictionary!
            for _, info in self.client_info_dict.items():
                mask = np.isin(labels.cpu().numpy(), info["label"])
                if mask.sum() > 0:
                    valid_t += 1
                    m_idx = torch.tensor(mask, device=self.device)
                    preds = info["model"].eval().to(self.device)(gen_imgs[m_idx])
                    total_ce += criterion_ce(preds, labels[m_idx])
                    if mask.sum() >= MIN_BN_SAMPLES: 
                        total_bn += self.get_bn_loss(info["model"], gen_imgs[m_idx])

            loss_g = loss_adv + (total_ce / max(1, valid_t)) + 0.1 * (total_bn / max(1, valid_t))
            loss_g.backward()
            self.optimizer_g.step()

    def get_bn_loss(self, teacher_model, gen_imgs):
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

        for hook in bn_hooks: 
            hook.remove()
        
        return loss_bn

    def train_global_classifier(self, seen_classes=None):
        self.global_model.train()
        self.global_generator.eval()
        
        seen_classes = self.get_seen_classes() if seen_classes is None else list(seen_classes)
        if not seen_classes:
            print("No seen classes yet. Skipping classifier training.")
            return
        seen_classes_tensor = torch.tensor(seen_classes, dtype=torch.long)

        samples_per_class = self.generated_samples_per_class
        num_seen_classes = len(seen_classes) 
        batch_size = getattr(self.args, 'batch_size', 64) 
        
        total_samples = num_seen_classes * samples_per_class
        all_labels = seen_classes_tensor.repeat_interleave(samples_per_class)
        
        shuffle_idx = torch.randperm(total_samples)
        all_labels = all_labels[shuffle_idx].to(self.device)
        num_batches = math.ceil(total_samples / batch_size)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)
            
            batch_labels = all_labels[start_idx:end_idx]
            current_batch_size = batch_labels.size(0) 
            
            z = torch.randn(current_batch_size, self.nz).to(self.device)
            
            with torch.no_grad():
                imgs = self.global_generator(z, batch_labels)
            
            self.optimizer_c.zero_grad()
            logits = self.global_model(imgs)
            
            loss = self.KD_loss(logits, batch_labels, T=2.0) 
            loss.backward()
            self.optimizer_c.step()

    def KD_loss(self, student_logits, labels, T=2.0):
        return F.cross_entropy(student_logits / T, labels)

    def visualize_synthetic_data(self, task):
        debug_dir = os.path.join("output_debug", self.args.dataset, f"task_{task}")
        os.makedirs(debug_dir, exist_ok=True)
        
        self.global_generator.eval()
        
        all_seen_classes = set()
        for client in self.clients:
            all_seen_classes.update(client.classes_so_far)
        
        all_seen_classes = sorted(list(all_seen_classes))
        
        if not all_seen_classes:
            print("No seen classes found to visualize.")
            return

        print(f"[Vis] Generating samples for {len(all_seen_classes)} seen classes: {all_seen_classes}")

        with torch.no_grad():
            labels = torch.tensor(all_seen_classes, dtype=torch.long).to(self.device)
            z = torch.randn(len(labels), self.nz).to(self.device)
            
            imgs = self.global_generator(z, labels)
            stats = self.global_generator.stats
            
            save_path = os.path.join(debug_dir, f"all_seen_classes_task_{task}.png")
            save_image(imgs, save_path, nrow=5 if self.dataset == 'CIFAR10' else 10 if self.dataset == 'CIFAR100' else 50, normalize=False)
            print(f"[Vis] Saved all-class grid to {save_path}")

    def receive_models(self):
        self.client_info_dict = {c.id: {"model": copy.deepcopy(c.model), "label": list(c.classes_so_far)} for c in self.selected_clients}
        super().receive_models()
        model = self.uploaded_models[0] if self.uploaded_models else self.global_model
        self.communication_accountant.record_uplink(model, len(self.uploaded_models))

    def send_models(self):
        for client in self.clients:
            client.set_parameters(self.global_model)
            client.set_generator_parameters(self.global_generator)
        self.communication_accountant.record_downlink(self.global_model, len(self.clients))
        self.communication_accountant.record_downlink(self.global_generator, len(self.clients))


    def get_seen_classes(self):
        all_seen_classes = set()
        for client in self.clients:
            if hasattr(client, 'classes_so_far'):
                all_seen_classes.update(client.classes_so_far)
        return sorted(list(all_seen_classes))

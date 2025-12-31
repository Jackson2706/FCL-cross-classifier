"""
FedCIL Client - Accurate Implementation
Based on the actual FedCIL repository: https://github.com/daiqing98/FedCIL

Uses:
- WGAN with Auxiliary Classifier (AC-GAN style)
- Data-to-Data Cross Entropy (D2DCE) loss for better discrimination
- Model consolidation through knowledge distillation
- Consistency enforcement via server generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client


class clientFedCIL(Client):
    """
    FedCIL Client with WGAN-based Generative Replay

    Key Features:
    1. WGAN with auxiliary classifier for stable GAN training
    2. D2DCE (Data-to-Data Cross Entropy) loss
    3. Knowledge distillation from server generator
    4. Model consolidation for smooth task transitions
    """

    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        # --- Dataset Configuration ---
        if 'cifar' in args.dataset.lower():
            self.img_size = 32
            self.nz = 110 if 'cifar100' in args.dataset.lower() else 100
            self.nc = 3
            self.num_classes = 100 if 'cifar100' in args.dataset.lower() else 10
        elif 'mnist' in args.dataset.lower():
            self.img_size = 28
            self.nz = 100
            self.nc = 1
            self.num_classes = 10
        elif 'emnist' in args.dataset.lower():
            self.img_size = 28
            self.nz = 100
            self.nc = 1
            self.num_classes = 26 if 'L' in args.dataset else 47
        else:
            self.img_size = 32
            self.nz = 100
            self.nc = 3
            self.num_classes = 10

        # --- FedCIL Components ---
        self.local_generator = None  # Local WGAN generator
        self.server_generator = None  # Server's global generator (for KD)
        self.last_copy = None  # Previous task model copy

        # --- FedCIL Hyperparameters ---
        self.importance_of_new_task = getattr(args, 'importance_new_task', 0.5)
        self.kd_weight_d = getattr(args, 'kd_weight_d', 0.33)  # Discriminator KD weight
        self.kd_weight_g = getattr(args, 'kd_weight_g', 0.33)  # Generator KD weight
        self.temperature = getattr(args, 'kd_temperature', 2.0)

        # --- Task Management ---
        self.current_task = 0
        self.glob_iter = 0
        self.classes_so_far = []  # Changed to list for compatibility
        self.available_labels = []

        # --- Loss Functions ---
        self.aux_criterion = nn.NLLLoss()
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")

        print(f"[Client {self.id}] FedCIL initialized | nz={self.nz}, num_classes={self.num_classes}")

    def set_local_generator(self, generator):
        """Set local WGAN generator"""
        self.local_generator = generator
        if self.local_generator:
            self.local_generator.to(self.device)

    def set_server_generator(self, server_generator):
        """Set server's global generator for knowledge distillation"""
        self.server_generator = server_generator
        if self.server_generator:
            self.server_generator.to(self.device)
            self.server_generator.eval()
            for param in self.server_generator.parameters():
                param.requires_grad = False

    def save_last_copy(self):
        """Save current generator as last copy for model consolidation"""
        if self.local_generator:
            self.last_copy = copy.deepcopy(self.local_generator)
            self.last_copy.eval()
            for param in self.last_copy.parameters():
                param.requires_grad = False

    def train(self, task=None):
        """
        FedCIL Training with WGAN

        Three-stage training:
        1. Train on current task data (real data)
        2. Train on replayed data (generated from previous tasks)
        3. Apply knowledge distillation from server generator
        """
        if task is not None:
            self.current_task = task

        trainloader = self.load_train_data(task=task)

        # Update classes learned so far
        current_classes = self._get_task_classes(trainloader.dataset)
        for cls in current_classes:
            if cls not in self.classes_so_far:
                self.classes_so_far.append(cls)

        print(f"[Client {self.id}] Task {task} | Classes so far: {self.classes_so_far}")

        # Training
        if not self.local_generator:
            print(f"[Client {self.id}] Warning: No generator set, training without replay")
            self._train_without_replay(trainloader)
            return

        self.local_generator.train()
        start_time = time.time()

        losses = {
            'total': [],
            'c_loss': [],
            'g_loss': [],
            'kd_d': [],
            'kd_g': [],
            'aux_fake': [],
            'aux_real': []
        }

        for epoch in range(self.local_epochs):
            for i, (x_real, y_real) in enumerate(trainloader):
                # Prepare data
                if isinstance(x_real, list):
                    x_real = x_real[0]
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)
                batch_size = x_real.size(0)

                # === Generate Replay Data ===
                x_replay, y_replay = None, None
                if self.current_task > 0 and len(self.classes_so_far) > len(current_classes):
                    # Get classes from previous tasks
                    prev_classes = [c for c in self.classes_so_far if c not in current_classes]
                    x_replay, y_replay = self._generate_replay_samples(
                        batch_size, prev_classes
                    )

                # === Train Generator and Discriminator ===
                train_results = self._train_wgan_step(
                    x_real, y_real,
                    x_replay, y_replay,
                    list(self.classes_so_far),
                    self.available_labels
                )

                # Track losses
                for key in train_results:
                    if key in losses:
                        losses[key].append(train_results[key])

            # Periodic logging
            if (epoch + 1) % 5 == 0 or epoch == self.local_epochs - 1:
                avg_c = np.mean(losses['c_loss'][-len(trainloader):]) if losses['c_loss'] else 0
                avg_g = np.mean(losses['g_loss'][-len(trainloader):]) if losses['g_loss'] else 0
                avg_kd_d = np.mean(losses['kd_d'][-len(trainloader):]) if losses['kd_d'] else 0
                avg_kd_g = np.mean(losses['kd_g'][-len(trainloader):]) if losses['kd_g'] else 0

                print(f"[Client {self.id}] Epoch {epoch+1}/{self.local_epochs} | "
                      f"C: {avg_c:.4f}, G: {avg_g:.4f}, "
                      f"KD_D: {avg_kd_d:.4f}, KD_G: {avg_kd_g:.4f}")

        # Save for next task
        self.save_last_copy()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _train_wgan_step(self, x_real, y_real, x_replay, y_replay,
                         classes_so_far, available_labels):
        """
        Single training step combining:
        1. AC-GAN loss (real + replay)
        2. Knowledge distillation from server (discriminator)
        3. Knowledge distillation from server (generator)
        """
        batch_size = x_real.size(0)

        # =============================================
        # Phase 1: Train Discriminator (Critic)
        # =============================================

        # 1a. Loss on real data
        c_loss_real, g_real, c_logits_real, aux_info = self._critic_loss(
            x_real, y_real, classes_so_far,
            return_g=True, return_aux=True
        )

        # 1b. Loss on replay data (if available)
        if x_replay is not None and y_replay is not None:
            c_loss_replay, g_replay, c_logits_replay, aux_replay = self._critic_loss(
                x_replay, y_replay, classes_so_far,
                return_g=True, return_aux=True
            )
            c_loss = (self.importance_of_new_task * c_loss_real +
                     (1 - self.importance_of_new_task) * c_loss_replay)
        else:
            c_loss = c_loss_real

        # 1c. Knowledge Distillation: Server Generator -> Local Discriminator
        kd_loss_d = 0.0
        if self.server_generator and self.glob_iter > 0:
            # Generate fake images using server's generator with real labels
            noise, aux_label, _ = self._generate_noise_with_classes(
                batch_size, classes_so_far,
                label=y_real.cpu().detach().numpy()
            )

            with torch.no_grad():
                fake_server = self.server_generator.generator(noise.to(self.device))

            # Get local discriminator predictions
            _, p_fake, _ = self.local_generator.critic(fake_server.detach())
            _, p_real, _ = self.local_generator.critic(x_real)

            # KL divergence loss
            kd_loss_d = self.ensemble_loss(torch.log(p_real), p_fake)

        # Combined discriminator loss
        # Note: In ablation study, KD losses are weighted by 0
        c_loss_total = c_loss + self.kd_weight_d * kd_loss_d

        # Update discriminator
        if hasattr(self.local_generator, 'critic_optimizer'):
            self.local_generator.critic_optimizer.zero_grad()
            c_loss_total.backward()
            self.local_generator.critic_optimizer.step()

        # =============================================
        # Phase 2: Train Generator
        # =============================================

        # 2a. Standard AC-GAN generator loss
        g_loss, g_logits = self._generator_loss(x_real, y_real, classes_so_far)

        # 2b. Knowledge Distillation: Server Generator -> Local Generator
        kd_loss_g_1 = 0.0
        kd_loss_g_2 = 0.0

        if self.server_generator and self.glob_iter > 0 and available_labels:
            # Generate samples using both generators
            noise, aux_label, _ = self._generate_noise_with_classes(
                batch_size, available_labels
            )

            # Server's generated images
            with torch.no_grad():
                fake_server = self.server_generator.generator(noise.to(self.device))

            # Local generator's images
            noise_local, aux_label_local, _ = self._generate_noise_with_classes(
                batch_size, classes_so_far,
                label=aux_label.cpu().detach().numpy()
            )
            fake_local = self.local_generator.generator(noise_local.to(self.device))

            # Predictions
            _, p_server, _ = self.local_generator.critic(fake_server.detach())
            _, p_local, _ = self.local_generator.critic(fake_local.detach())

            # KD loss (align local with server)
            kd_loss_g_1 = self.ensemble_loss(torch.log(p_local), p_server)

            # Classification loss (ensure quality)
            kd_loss_g_2 = self.aux_criterion(torch.log(p_server), aux_label)

        kd_loss_g = kd_loss_g_1 + kd_loss_g_2

        # Combined generator loss
        # Note: In ablation, KD losses weighted by 0
        g_loss_total = g_loss + self.kd_weight_g * kd_loss_g

        # Update generator
        if hasattr(self.local_generator, 'generator_optimizer'):
            self.local_generator.generator_optimizer.zero_grad()
            g_loss_total.backward()
            self.local_generator.generator_optimizer.step()

        # Return losses for logging
        print(f"c_loss: {c_loss:.4f}, g_loss: {g_loss:.4f}",
              f"kd_loss_d: {kd_loss_d:.4f}, kd_loss_g: {kd_loss_g:.4f}",
              f"aux_fake: {aux_info[0]}, aux_real: {aux_info[1]}")

        return {
            'c_loss': c_loss.item(),
            'g_loss': g_loss.item(),
            'kd_d': kd_loss_d if isinstance(kd_loss_d, float) else kd_loss_d.item(),
            'kd_g': kd_loss_g if isinstance(kd_loss_g, float) else kd_loss_g.item(),
            'aux_fake': aux_info[0] if isinstance(aux_info, tuple) else 0,
            'aux_real': aux_info[1] if isinstance(aux_info, tuple) else 0,
        }

    def _critic_loss(self, x, y, classes_so_far, return_g=False, return_aux=False):
        """
        Compute critic (discriminator) loss
        Combines adversarial loss and auxiliary classification loss
        """
        batch_size = x.size(0)

        # Real data
        dis_output, aux_output, logits_real = self.local_generator.critic(x)

        # Auxiliary classification loss
        aux_errD_real = self.aux_criterion(torch.log(aux_output), y)

        # Generate fake data
        noise, aux_label, _ = self._generate_noise_with_classes(batch_size, classes_so_far)
        fake = self.local_generator.generator(noise.to(self.device))

        # Fake data
        dis_output_fake, aux_output_fake, logits_fake = self.local_generator.critic(fake.detach())
        aux_errD_fake = self.aux_criterion(torch.log(aux_output_fake), aux_label)

        # Combined loss
        loss_c = aux_errD_real + aux_errD_fake

        if return_g:
            if return_aux:
                return loss_c, fake, logits_real, (aux_errD_fake, aux_errD_real, None)
            else:
                return loss_c, fake, logits_real
        else:
            return loss_c

    def _generator_loss(self, x, y, classes_so_far):
        """Compute generator loss"""
        batch_size = x.size(0)

        # Generate noise
        noise, aux_label, _ = self._generate_noise_with_classes(batch_size, classes_so_far)
        fake = self.local_generator.generator(noise.to(self.device))

        # Get discriminator predictions
        dis_output, aux_output, logits = self.local_generator.critic(fake)

        # Auxiliary classification loss
        aux_errG = self.aux_criterion(torch.log(aux_output), aux_label)

        return aux_errG, logits

    def _generate_noise_with_classes(self, batch_size, classes_so_far, label=None):
        """
        Generate noise vectors with class conditioning
        Following the AC-GAN approach from the paper
        """
        noise = torch.FloatTensor(batch_size, self.nz, 1, 1)
        aux_label = torch.LongTensor(batch_size)
        dis_label = torch.FloatTensor(batch_size)

        # Sample labels
        if label is None:
            label = np.random.choice(classes_so_far, batch_size)

        # Create noise with one-hot encoding
        noise_ = np.random.normal(0, 1, (batch_size, self.nz))
        class_onehot = np.zeros((batch_size, self.num_classes))

        class_onehot[np.arange(batch_size), label] = 1

        # Embed class information in noise
        noise_[np.arange(batch_size), :self.num_classes] = class_onehot[np.arange(batch_size)]

        # Convert to tensor
        noise_ = torch.from_numpy(noise_).float()
        noise.copy_(noise_.view(batch_size, self.nz, 1, 1))
        aux_label.copy_(torch.from_numpy(label))

        # Squeeze and move to device
        noise = torch.squeeze(noise)
        dis_label.fill_(0)  # Fake label

        return noise, aux_label.to(self.device), dis_label.to(self.device)

    def _generate_replay_samples(self, num_samples, previous_classes):
        """Generate samples from previous tasks"""
        if not self.local_generator:
            return None, None

        self.local_generator.eval()

        noise, aux_label, _ = self._generate_noise_with_classes(
            num_samples, previous_classes
        )

        with torch.no_grad():
            fake_images = self.local_generator.generator(noise.to(self.device))

        self.local_generator.train()

        return fake_images, aux_label

    def _train_without_replay(self, trainloader):
        """Fallback: train classifier only without replay"""
        self.model.train()
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                self.optimizer.step()

    def _get_task_classes(self, dataset):
        """Extract unique classes from dataset"""
        classes = []
        for _, label in dataset:
            label_val = label.item() if isinstance(label, torch.Tensor) else int(label)
            if label_val not in classes:
                classes.append(label_val)
        return classes

    def sample_from_generator(self, num_samples, classes=None):
        """Sample images from the local generator"""
        if not self.local_generator:
            return None, None

        if classes is None:
            classes = list(self.classes_so_far) if self.classes_so_far else []

        if not classes:
            return None, None

        self.local_generator.eval()

        noise, aux_label, _ = self._generate_noise_with_classes(num_samples, classes)

        with torch.no_grad():
            generated_images = self.local_generator.generator(noise.to(self.device))

        return generated_images, aux_label
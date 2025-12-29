"""
FedCIL Client Implementation

Based on "Better Generative Replay for Continual Federated Learning" (ICLR 2023)
Paper: https://arxiv.org/abs/2302.13001

Key Components:
1. AC-GAN for generative replay
2. Model Consolidation - stabilizes training with non-IID data
3. Consistency Enforcement - maintains performance across clients
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
    FedCIL Client with Generative Replay

    Main Features:
    - AC-GAN (Auxiliary Classifier GAN) for generating old task samples
    - Model consolidation to stabilize federated training
    - Consistency enforcement between local and global models
    """

    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        # --- Dataset Configuration ---
        if 'cifar100' in args.dataset.lower():
            self.img_size = 32
            self.nz = 256
            self.nc = 3
        elif 'imagenet' in args.dataset.lower():
            self.img_size = 64 if '100' in args.dataset else 224
            self.nz = 256
            self.nc = 3
        else:
            self.img_size = 32
            self.nz = 100
            self.nc = 3

        # --- FedCIL Hyperparameters ---
        self.lambda_cons = getattr(args, 'lambda_cons', 1.0)  # Consistency loss weight
        self.lambda_gen = getattr(args, 'lambda_gen', 0.5)  # Generated sample loss weight
        self.T_kd = getattr(args, 'T_kd', 2.0)  # Temperature for knowledge distillation
        self.replay_ratio = getattr(args, 'replay_ratio', 1.0)

        # --- AC-GAN Components ---
        self.generator = None
        self.discriminator = None

        # --- Previous Models (for consolidation) ---
        self.prev_model = None  # Previous task's model
        self.global_model_copy = None  # Server's global model for consistency

        # --- Task Management ---
        self.current_task = 0
        self.learned_classes = set()
        self.current_classes = set()

        print(f"[Client {self.id}] FedCIL initialized")

    def set_generator(self, generator):
        """Receive AC-GAN generator from server"""
        self.generator = generator
        if self.generator:
            self.generator.to(self.device)
            # Freeze generator during client training
            self.generator.eval()
            for param in self.generator.parameters():
                param.requires_grad = False

    def set_discriminator(self, discriminator):
        """Receive AC-GAN discriminator from server"""
        self.discriminator = discriminator
        if self.discriminator:
            self.discriminator.to(self.device)

    def set_global_model_copy(self, global_model):
        """
        Store copy of global model for consistency enforcement
        This helps stabilize training with non-IID data
        """
        self.global_model_copy = copy.deepcopy(global_model)
        self.global_model_copy.eval()
        for param in self.global_model_copy.parameters():
            param.requires_grad = False
        self.global_model_copy.to(self.device)

    def save_prev_model(self):
        """Save current model as previous for model consolidation"""
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.eval()
        for param in self.prev_model.parameters():
            param.requires_grad = False

    def generate_replay_samples(self, num_samples, previous_classes):
        """
        Generate samples from previous tasks using AC-GAN

        Args:
            num_samples: Number of samples to generate
            previous_classes: List of classes from previous tasks

        Returns:
            Generated images and labels
        """
        if not self.generator or not previous_classes:
            return None, None

        self.generator.eval()

        # Sample classes uniformly
        samples_per_class = max(1, num_samples // len(previous_classes))

        gen_images = []
        gen_labels = []

        with torch.no_grad():
            for class_id in previous_classes:
                # Sample latent vectors
                z = torch.randn(samples_per_class, self.nz).to(self.device)
                # Create labels
                labels = torch.full((samples_per_class,), class_id, dtype=torch.long).to(self.device)

                # Generate images
                fake_images = self.generator(z, labels)

                gen_images.append(fake_images)
                gen_labels.append(labels)

        gen_images = torch.cat(gen_images, dim=0)
        gen_labels = torch.cat(gen_labels, dim=0)

        # Shuffle
        perm = torch.randperm(gen_images.size(0))
        gen_images = gen_images[perm][:num_samples]
        gen_labels = gen_labels[perm][:num_samples]

        return gen_images, gen_labels

    def consistency_loss(self, images, use_kd=True):
        """
        Consistency Enforcement Loss

        Ensures local model predictions are consistent with global model
        This addresses the instability from non-IID data distribution

        Args:
            images: Input images
            use_kd: Use knowledge distillation (soft targets)

        Returns:
            Consistency loss value
        """
        if self.global_model_copy is None:
            return torch.tensor(0.0).to(self.device)

        self.model.train()
        self.global_model_copy.eval()

        # Local model predictions
        local_logits = self.model(images)

        # Global model predictions (teacher)
        with torch.no_grad():
            global_logits = self.global_model_copy(images)

        if use_kd:
            # Knowledge distillation with temperature
            loss = F.kl_div(
                F.log_softmax(local_logits / self.T_kd, dim=1),
                F.softmax(global_logits / self.T_kd, dim=1),
                reduction='batchmean'
            ) * (self.T_kd ** 2)
        else:
            # MSE loss on logits
            loss = F.mse_loss(local_logits, global_logits)

        return loss

    def consolidation_loss(self, gen_images):
        """
        Model Consolidation Loss

        Prevents forgetting by ensuring predictions on generated samples
        match the previous model's predictions

        Args:
            gen_images: Generated images from previous tasks

        Returns:
            Consolidation loss value
        """
        if self.prev_model is None or gen_images is None:
            return torch.tensor(0.0).to(self.device)

        self.model.train()
        self.prev_model.eval()

        # Current model predictions
        curr_logits = self.model(gen_images)

        # Previous model predictions (soft targets)
        with torch.no_grad():
            prev_logits = self.prev_model(gen_images)

        # KL divergence loss
        loss = F.kl_div(
            F.log_softmax(curr_logits / self.T_kd, dim=1),
            F.softmax(prev_logits / self.T_kd, dim=1),
            reduction='batchmean'
        ) * (self.T_kd ** 2)

        return loss

    def train(self, task=None):
        """
        FedCIL Training with:
        1. Real data from current task
        2. Generated data from previous tasks (replay)
        3. Consistency enforcement with global model
        4. Model consolidation with previous model
        """
        if task is not None:
            self.current_task = task

        trainloader = self.load_train_data(task=task)

        # Update learned classes
        self.current_classes = self._get_task_classes(trainloader.dataset)
        previous_classes = list(self.learned_classes - self.current_classes)
        self.learned_classes.update(self.current_classes)

        print(f"[Client {self.id}] Task {task} - Current classes: {self.current_classes}, "
              f"Previous classes: {len(previous_classes)}")

        # Save previous model for consolidation
        if self.current_task > 0:
            self.save_prev_model()

        # Training
        self.model.train()
        start_time = time.time()

        losses = {
            'total': [],
            'ce_real': [],
            'ce_gen': [],
            'consistency': [],
            'consolidation': []
        }

        for epoch in range(self.local_epochs):
            for i, (x_real, y_real) in enumerate(trainloader):
                # Prepare real data
                if isinstance(x_real, list):
                    x_real = x_real[0]
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)
                batch_size = x_real.size(0)

                self.optimizer.zero_grad()

                # === 1. Loss on Real Data (Current Task) ===
                output_real = self.model(x_real)
                loss_ce_real = F.cross_entropy(output_real, y_real)

                # === 2. Consistency Enforcement Loss ===
                # Stabilizes training with non-IID data
                loss_consistency = self.consistency_loss(x_real, use_kd=True)

                # === 3. Generative Replay (if previous tasks exist) ===
                loss_ce_gen = torch.tensor(0.0).to(self.device)
                loss_consolidation = torch.tensor(0.0).to(self.device)

                if self.generator and len(previous_classes) > 0:
                    # Generate samples from previous tasks
                    num_gen = int(batch_size * self.replay_ratio)
                    gen_images, gen_labels = self.generate_replay_samples(
                        num_gen, previous_classes
                    )

                    if gen_images is not None:
                        # Classification loss on generated samples
                        output_gen = self.model(gen_images)
                        loss_ce_gen = F.cross_entropy(output_gen, gen_labels)

                        # Model consolidation loss
                        loss_consolidation = self.consolidation_loss(gen_images)

                # === 4. Combined Loss ===
                loss = (loss_ce_real +
                        self.lambda_gen * loss_ce_gen +
                        self.lambda_cons * loss_consistency +
                        loss_consolidation)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

                # Track losses
                losses['total'].append(loss.item())
                losses['ce_real'].append(loss_ce_real.item())
                losses['ce_gen'].append(loss_ce_gen.item())
                losses['consistency'].append(loss_consistency.item())
                losses['consolidation'].append(loss_consolidation.item())

            # Learning rate scheduling
            if self.learning_rate_scheduler:
                self.learning_rate_scheduler.step()

            # Periodic logging
            if (epoch + 1) % 5 == 0 or epoch == self.local_epochs - 1:
                avg_total = np.mean(losses['total'][-len(trainloader):])
                avg_real = np.mean(losses['ce_real'][-len(trainloader):])
                avg_gen = np.mean(losses['ce_gen'][-len(trainloader):])
                avg_cons = np.mean(losses['consistency'][-len(trainloader):])
                avg_consol = np.mean(losses['consolidation'][-len(trainloader):])
                lr = self.optimizer.param_groups[0]['lr']

                print(f"[Client {self.id}] Epoch {epoch + 1}/{self.local_epochs} | "
                      f"Total: {avg_total:.4f} (Real: {avg_real:.4f}, Gen: {avg_gen:.4f}, "
                      f"Cons: {avg_cons:.4f}, Consol: {avg_consol:.4f}) | LR: {lr:.6f}")

        # Final statistics
        print(f"[Client {self.id}] Task {task} complete | "
              f"Avg Loss: {np.mean(losses['total']):.4f}")

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _get_task_classes(self, dataset):
        """Extract unique classes from dataset"""
        classes = set()
        for _, label in dataset:
            if isinstance(label, torch.Tensor):
                classes.add(label.item())
            else:
                classes.add(int(label))
        return classes

    def evaluate_forgetting(self, all_test_loaders):
        """
        Measure catastrophic forgetting across all tasks

        Args:
            all_test_loaders: Dict mapping task_id -> test_loader

        Returns:
            Dictionary of accuracies per task
        """
        self.model.eval()
        task_accuracies = {}

        for task_id, test_loader in all_test_loaders.items():
            correct = 0
            total = 0

            with torch.no_grad():
                for x, y in test_loader:
                    if isinstance(x, list):
                        x = x[0]
                    x, y = x.to(self.device), y.to(self.device)

                    outputs = self.model(x)
                    _, predicted = torch.max(outputs, 1)

                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            acc = 100.0 * correct / total if total > 0 else 0.0
            task_accuracies[task_id] = acc

        # Compute average and forgetting
        if task_accuracies:
            avg_acc = np.mean(list(task_accuracies.values()))
            print(f"\n[Client {self.id}] FedCIL Evaluation Results:")
            for task_id, acc in task_accuracies.items():
                print(f"  Task {task_id}: {acc:.2f}%")
            print(f"  Average: {avg_acc:.2f}%\n")

        return task_accuracies
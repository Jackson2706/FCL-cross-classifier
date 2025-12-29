import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client


class clientDGR(Client):
    """
    Deep Generative Replay Client
    Based on "Continual Learning with Deep Generative Replay" (NIPS 2017)

    Uses a Scholar model consisting of:
    - Generator: Creates synthetic samples from previous tasks
    - Solver: Classifier that learns current task while maintaining old knowledge
    """

    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        # --- Dataset Configuration ---
        if 'cifar100' in args.dataset.lower():
            self.img_size = 32
            self.nz = 256  # Latent dimension
            self.nc = 3  # Number of channels
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)
        elif 'imagenet' in args.dataset.lower():
            self.img_size = 64 if '100' in args.dataset else 224
            self.nz = 256
            self.nc = 3
        else:
            self.img_size = 32
            self.nz = 100
            self.nc = 3

        # --- DGR Hyperparameters ---
        self.replay_enabled = getattr(args, 'dgr_replay_enabled', True)
        self.replay_ratio = getattr(args, 'dgr_replay_ratio', 1.0)  # Ratio of replayed to real samples
        self.generator_lr = getattr(args, 'dgr_generator_lr', 0.0002)
        self.generator_updates = getattr(args, 'dgr_generator_updates', 5)  # Generator updates per solver update
        self.sample_importance = getattr(args, 'dgr_sample_importance', 0.5)  # Balance between old and new tasks

        # --- Generator (GAN-based) ---
        self.generator = None
        self.generator_optimizer = None
        self.discriminator = None  # For training generator if needed

        # --- Previous Task Models (for distillation) ---
        self.previous_model = None
        self.previous_generator = None

        # --- Training Statistics ---
        self.task_classes = {}  # Maps task_id -> set of classes
        self.current_task = 0

    def set_generator(self, generator):
        """
        Set generator from server or initialize locally
        """
        self.generator = generator
        if self.generator:
            self.generator.to(self.device)
            self.generator_optimizer = torch.optim.Adam(
                self.generator.parameters(),
                lr=self.generator_lr,
                betas=(0.5, 0.999)
            )
            # Freeze initially - will be unfrozen during training
            self.freeze_generator()

    def freeze_generator(self):
        """Freeze generator parameters"""
        if self.generator:
            self.generator.eval()
            for param in self.generator.parameters():
                param.requires_grad = False

    def unfreeze_generator(self):
        """Unfreeze generator for training"""
        if self.generator:
            self.generator.train()
            for param in self.generator.parameters():
                param.requires_grad = True

    def save_previous_model(self):
        """
        Save current model as previous model for knowledge retention
        """
        self.previous_model = copy.deepcopy(self.model)
        self.previous_model.eval()
        for param in self.previous_model.parameters():
            param.requires_grad = False

        if self.generator:
            self.previous_generator = copy.deepcopy(self.generator)
            self.previous_generator.eval()
            for param in self.previous_generator.parameters():
                param.requires_grad = False

    def generate_replay_samples(self, num_samples, task_classes=None):
        """
        Generate synthetic samples from previous tasks using the generator

        Args:
            num_samples: Number of samples to generate
            task_classes: List of class labels to generate from (if None, use all previous classes)

        Returns:
            Generated images and their labels
        """
        if not self.generator or not self.replay_enabled:
            return None, None

        self.generator.eval()

        if task_classes is None:
            # Get all classes from previous tasks
            all_previous_classes = []
            for task_id in range(self.current_task):
                if task_id in self.task_classes:
                    all_previous_classes.extend(self.task_classes[task_id])

            if not all_previous_classes:
                return None, None

            task_classes = all_previous_classes

        # Generate samples
        generated_images = []
        generated_labels = []

        samples_per_class = max(1, num_samples // len(task_classes))

        with torch.no_grad():
            for class_id in task_classes:
                z = torch.randn(samples_per_class, self.nz).to(self.device)
                class_labels = torch.full((samples_per_class,), class_id, dtype=torch.long).to(self.device)

                # Generate images
                if hasattr(self.generator, 'generate'):
                    fake_images = self.generator.generate(z, class_labels)
                else:
                    fake_images = self.generator(z, class_labels)

                generated_images.append(fake_images)
                generated_labels.append(class_labels)

        if generated_images:
            generated_images = torch.cat(generated_images, dim=0)
            generated_labels = torch.cat(generated_labels, dim=0)
            return generated_images, generated_labels

        return None, None

    def get_replayed_predictions(self, generated_images):
        """
        Get predictions from previous model on generated samples (for training current model)
        """
        if self.previous_model is None or generated_images is None:
            return None

        self.previous_model.eval()
        with torch.no_grad():
            predictions = self.previous_model(generated_images)

        return predictions

    def train_generator_step(self, real_images, real_labels):
        """
        Train generator using adversarial loss and solver feedback
        This implements the Scholar's generator training
        """
        if not self.generator or not self.replay_enabled:
            return 0.0

        self.unfreeze_generator()
        batch_size = real_images.size(0)

        # Generate fake samples
        z = torch.randn(batch_size, self.nz).to(self.device)
        fake_labels = real_labels  # Generate same class distribution as real

        fake_images = self.generator(z, fake_labels)

        # Get solver predictions on fake images
        fake_predictions = self.model(fake_images)

        # Generator loss: fool solver to predict correct labels
        # This encourages generator to create realistic, correctly-labeled samples
        gen_loss = F.cross_entropy(fake_predictions, fake_labels)

        # Optional: Add discriminator loss if using GAN
        if hasattr(self, 'discriminator') and self.discriminator is not None:
            fake_validity = self.discriminator(fake_images)
            real_validity = self.discriminator(real_images.detach())

            # GAN loss (trying to fool discriminator)
            gan_loss = -torch.mean(fake_validity)
            gen_loss = gen_loss + 0.1 * gan_loss

        # Update generator
        self.generator_optimizer.zero_grad()
        gen_loss.backward()
        self.generator_optimizer.step()

        self.freeze_generator()

        return gen_loss.item()

    def compute_dgr_loss(self, output, y, replayed_output=None, replayed_targets=None):
        """
        Compute Deep Generative Replay loss
        Balances learning new task with maintaining old knowledge

        Args:
            output: Model predictions on real data
            y: Real labels
            replayed_output: Model predictions on generated/replayed data
            replayed_targets: Previous model's predictions on generated data (soft targets)
        """
        # Loss on current task (real data)
        loss_current = F.cross_entropy(output, y)

        # If no replay data, return current loss only
        if replayed_output is None or replayed_targets is None:
            return loss_current, loss_current.item(), 0.0

        # Loss on replayed data (knowledge distillation from previous model)
        # Use KL divergence to match previous model's predictions
        loss_replay = F.kl_div(
            F.log_softmax(replayed_output, dim=1),
            F.softmax(replayed_targets, dim=1),
            reduction='batchmean'
        )

        # Combine losses with importance weighting
        # sample_importance controls balance between old and new knowledge
        total_loss = (1 - self.sample_importance) * loss_current + self.sample_importance * loss_replay

        return total_loss, loss_current.item(), loss_replay.item()

    def train(self, task=None):
        """
        Train with Deep Generative Replay

        Key differences from standard training:
        1. Generate synthetic samples from previous tasks
        2. Train on mixture of real (current task) and generated (previous tasks) data
        3. Update generator to maintain quality of replayed samples
        4. Use knowledge distillation to preserve old task performance
        """
        if task is not None:
            self.current_task = task

        trainloader = self.load_train_data(task=task)

        # Track classes in current task
        if task not in self.task_classes:
            self.task_classes[task] = set()
            for _, y in trainloader.dataset:
                if isinstance(y, torch.Tensor):
                    self.task_classes[task].add(y.item())
                else:
                    self.task_classes[task].add(int(y))

        print(f"[Client {self.id}] Task {task} classes: {self.task_classes[task]}")

        # Prepare model for training
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        start_time = time.time()

        # Training statistics
        total_loss = []
        current_task_losses = []
        replay_losses = []
        generator_losses = []

        for epoch in range(self.local_epochs):
            epoch_loss = []
            epoch_current = []
            epoch_replay = []
            epoch_gen = []

            for i, (x, y) in enumerate(trainloader):
                # Prepare real data
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device)
                y = y.to(self.device)
                batch_size = x.size(0)

                # === Train Generator (multiple updates per solver update) ===
                if self.generator and self.replay_enabled and epoch > 0:  # Start after first epoch
                    for _ in range(self.generator_updates):
                        gen_loss = self.train_generator_step(x, y)
                        epoch_gen.append(gen_loss)

                # === Generate Replay Samples ===
                replayed_images = None
                replayed_predictions = None

                if self.replay_enabled and self.current_task > 0:
                    num_replay = int(batch_size * self.replay_ratio)
                    replayed_images, replayed_labels = self.generate_replay_samples(num_replay)

                    if replayed_images is not None:
                        # Get previous model's predictions on generated samples
                        replayed_predictions = self.get_replayed_predictions(replayed_images)

                # === Train Solver (Classifier) ===
                self.optimizer.zero_grad()

                # Forward pass on real data
                output = self.model(x)

                # Forward pass on replayed data
                replayed_output = None
                if replayed_images is not None:
                    replayed_output = self.model(replayed_images)

                # Compute combined loss
                loss, loss_current, loss_replay = self.compute_dgr_loss(
                    output, y, replayed_output, replayed_predictions
                )

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

                # Statistics
                epoch_loss.append(loss.item())
                epoch_current.append(loss_current)
                epoch_replay.append(loss_replay)

            # Epoch statistics
            avg_loss = np.mean(epoch_loss) if epoch_loss else 0
            avg_current = np.mean(epoch_current) if epoch_current else 0
            avg_replay = np.mean(epoch_replay) if epoch_replay else 0
            avg_gen = np.mean(epoch_gen) if epoch_gen else 0

            total_loss.extend(epoch_loss)
            current_task_losses.extend(epoch_current)
            replay_losses.extend(epoch_replay)
            generator_losses.extend(epoch_gen)

            if epoch % 5 == 0 or epoch == self.local_epochs - 1:
                print(f"[Client {self.id}] Task {task} Epoch {epoch + 1}/{self.local_epochs} | "
                      f"Total: {avg_loss:.4f} | Current: {avg_current:.4f} | "
                      f"Replay: {avg_replay:.4f} | Gen: {avg_gen:.4f}")

            # Learning rate scheduling
            if self.learning_rate_scheduler:
                self.learning_rate_scheduler.step()

        # Save current model for next task
        self.save_previous_model()

        # Final logging
        current_lr = self.optimizer.param_groups[0]['lr']
        final_loss = np.mean(total_loss) if total_loss else 0
        final_current = np.mean(current_task_losses) if current_task_losses else 0
        final_replay = np.mean(replay_losses) if replay_losses else 0

        print(f"\n[Client {self.id}] Task {task} Training Complete:")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Current Task Loss: {final_current:.4f}")
        print(f"  Replay Loss: {final_replay:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Replay Enabled: {self.replay_enabled}")
        print(f"  Tasks Learned: {len(self.task_classes)}\n")

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _count_samples_per_class(self, dataset):
        """Count number of samples per class in dataset"""
        class_counts = {}
        for _, label in dataset:
            label = int(label) if not isinstance(label, int) else label
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts

    def get_feature_embeddings(self, model=None, task_id=None, num_samples=None):
        """Extract feature embeddings for visualization (e.g., t-SNE)"""
        if task_id is None:
            task_id = self.current_task

        target_model = model if model is not None else self.model
        dataloader = self.load_test_data(task=task_id, batch_size=self.batch_size)

        target_model.eval()
        target_model.to(self.device)

        features_list = []
        labels_list = []
        count = 0

        with torch.no_grad():
            for x, y in dataloader:
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)

                # Extract features from penultimate layer
                if hasattr(target_model, 'base'):
                    feats = target_model.base(x)
                else:
                    # Get features from layer before classifier
                    modules = list(target_model.children())[:-1]
                    feats = nn.Sequential(*modules)(x).view(x.size(0), -1)

                features_list.append(feats.detach().cpu())
                labels_list.append(y.detach().cpu())

                count += len(y)
                if num_samples and count >= num_samples:
                    break

        if features_list:
            return torch.cat(features_list), torch.cat(labels_list)
        return torch.tensor([]), torch.tensor([])

    def evaluate_on_all_tasks(self, verbose=True):
        """
        Evaluate model performance on all learned tasks
        Useful for measuring catastrophic forgetting
        """
        self.model.eval()
        task_accuracies = {}

        for task_id in self.task_classes.keys():
            correct = 0
            total = 0

            test_loader = self.load_test_data(task=task_id, batch_size=self.batch_size)

            with torch.no_grad():
                for x, y in test_loader:
                    if isinstance(x, list):
                        x = x[0]
                    x, y = x.to(self.device), y.to(self.device)

                    outputs = self.model(x)
                    _, predicted = torch.max(outputs.data, 1)

                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            accuracy = 100 * correct / total if total > 0 else 0
            task_accuracies[task_id] = accuracy

            if verbose:
                print(f"[Client {self.id}] Task {task_id} Accuracy: {accuracy:.2f}%")

        # Compute average accuracy across all tasks
        if task_accuracies:
            avg_accuracy = np.mean(list(task_accuracies.values()))
            if verbose:
                print(f"[Client {self.id}] Average Accuracy: {avg_accuracy:.2f}%")
            return avg_accuracy, task_accuracies

        return 0.0, {}
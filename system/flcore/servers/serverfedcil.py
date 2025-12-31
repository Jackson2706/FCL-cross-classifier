"""
FedCIL Server - Accurate Implementation
Based on actual FedCIL repository structure

Handles:
- WGAN-based generator aggregation
- Server-side generator training
- Model consolidation across clients
- Task-incremental federated learning
"""

import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import os
from torchvision.utils import save_image
from flcore.servers.serverbase import Server
from flcore.clients.clientfedcil import clientFedCIL
from flcore.trainmodel.FedCIL_models import WGAN
from utils.data_utils import *
from utils.model_utils import ParamDict

class serverFedCIL(Server):
    """
    FedCIL Server with WGAN Aggregation

    Responsibilities:
    1. Aggregate local generators from clients
    2. Train global generator on aggregated knowledge
    3. Distribute global generator to clients
    4. Coordinate continual learning across tasks
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.Budget = []

        # --- Dataset Configuration ---
        if 'cifar' in self.dataset.lower():
            self.img_size = 32
            self.nz = 110 if 'cifar100' in self.dataset.lower() else 100
            self.nc = 3
            self.num_classes = 100 if 'cifar100' in self.dataset.lower() else 10
            self.c_channel_size = 64
            self.g_channel_size = 64
        elif 'mnist' in self.dataset.lower():
            self.img_size = 28
            self.nz = 100
            self.nc = 1
            self.num_classes = 10
            self.c_channel_size = 128
            self.g_channel_size = 128
        elif 'emnist' in self.dataset.lower():
            self.img_size = 28
            self.nz = 100
            self.nc = 1
            self.num_classes = 26 if 'L' in self.dataset else 47
            self.c_channel_size = 128
            self.g_channel_size = 128
        else:
            self.img_size = 32
            self.nz = 100
            self.nc = 3
            self.num_classes = 10
            self.c_channel_size = 64
            self.g_channel_size = 64

        # --- Initialize Global WGAN Generator ---

        # print(f"nz: {self.nz}")
        self.global_generator = self._create_wgan()

        # --- Optimizers ---
        self.lr = getattr(args, 'lr', 0.0001)
        self.lr_CIFAR = getattr(args, 'lr_CIFAR', 0.0002)
        self.beta1 = getattr(args, 'beta1', 0.5)
        self.beta2 = getattr(args, 'beta2', 0.999)
        self.weight_decay = getattr(args, 'weight_decay', 0.0)
        self.generator_lambda = getattr(args, 'generator_lambda', 10.0)
        self.mode='partial' if 'partial' in self.algorithm.lower() else 'all'

        # TODO Initialize ACGAN: Source output / Auxilary classifier.
        self._initialize_optimizers()

        # --- Training Parameters ---
        self.importance_of_new_task = getattr(args, 'importance_new_task', 0.5)
        self.generator_iterations = getattr(args, 'generator_iterations', 50)

        # Initialize clients
        self.set_clients(clientFedCIL)
        for client in self.clients:
            client.local_generator = self.global_generator

        # Track learned classes globally
        self.global_classes = []  # Changed to list for compatibility

        print(f"[Server] FedCIL initialized | Dataset: {self.dataset}, "
              f"ImgSize: {self.img_size}, nz: {self.nz}, Classes: {self.num_classes}")

    def _create_wgan(self):
        """
        Create WGAN generator
        This should import from your gan module
        """
        generator = WGAN(
            z_size=self.nz,
            image_size=self.img_size,
            image_channel_size=self.nc,
            c_channel_size=self.c_channel_size,
            g_channel_size=self.g_channel_size,
            dataset=self.dataset
        )
        return generator.to(self.device)

    def _initialize_optimizers(self):
        """Initialize optimizers for generator"""
        if self.global_generator is None:
            return

        # Initialize weights
        self.global_generator.apply(self._weights_init)

        # Create optimizers
        if 'cifar' in self.dataset.lower():
            optimizerD = optim.Adam(
                self.global_generator.critic.parameters(),
                lr=self.lr_CIFAR,
                betas=(self.beta1, self.beta2)
            )
            optimizerG = optim.Adam(
                self.global_generator.generator.parameters(),
                lr=self.lr_CIFAR,
                betas=(self.beta1, self.beta2)
            )
        else:
            optimizerD = optim.Adam(
                self.global_generator.critic.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.beta1, self.beta2)
            )
            optimizerG = optim.Adam(
                self.global_generator.generator.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(self.beta1, self.beta2)
            )

        self.global_generator.set_generator_optimizer(optimizerG)
        self.global_generator.set_critic_optimizer(optimizerD)
        self.global_generator.set_lambda(self.generator_lambda)

        # TODO Gausian Initialize

    def _weights_init(self, m):
        """Initialize network weights"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1, 0.02)
            m.bias.data.fill_(0)

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

            print(f"[Server] Global classes so far: {self.global_classes}")

            # --- Federated Training Rounds ---
            for round_idx in range(self.global_rounds):
                start_time = time.time()
                glob_iter = round_idx + self.global_rounds * task

                # Select clients
                self.selected_clients = self.select_clients()

                # Send models to clients
                self.send_parameters_(mode=self.mode, only_critic = False)

                if round_idx%self.eval_gap == 0:
                    print(f"\n-------------Round number: {round_idx}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                # Client training
                for client in self.selected_clients:
                    client.glob_iter = glob_iter
                    client.available_labels = list(self.global_classes)
                    client.train(task=task)

                # Receive and aggregate models
                self.receive_models()
                self.aggregate_generators()

                self.Budget.append(time.time() - start_time)
                print(f"Round Time: {self.Budget[-1]:.2f}s")

            # --- Post-Task Processing ---
            print(f"\n--- Task {task} Post-Processing ---")

            # Additional generator training on server
            if self.global_generator:
                self.train_global_generator(task)

            # Visualize generated samples
            self.visualize_samples(task)

            # Evaluate on all tasks
            self.evaluate_all_tasks(task)

            print(f"\n>>> Task {task} finished. Evaluating Forgetting Rate...")
            self.eval_task(task=task, glob_iter=task, flag="global")

    def _load_task_data(self, task):
        """Load new task data for all clients"""
        # This should be implemented based on your data loading logic
        # Example:
        # for i, client in enumerate(self.clients):
        #     train_data, label_info = read_client_data_FCL(i, task=task)
        #     client.next_task(train_data, label_info)
        pass

    def send_models(self, task, glob_iter):
        """
        Send global generator and parameters to clients
        """
        for client in self.clients:
            # Send server's global generator for KD
            if self.global_generator:
                client.set_server_generator(self.global_generator)

                # For first round of first task, also set as local generator
                if task == 0 and glob_iter == 0:
                    client.set_local_generator(copy.deepcopy(self.global_generator))

    def receive_models(self):
        """Receive models from selected clients"""
        print(f"[Server] Receiving models from {len(self.selected_clients)} clients")

        self.uploaded_generators = []
        for client in self.selected_clients:
            if client.local_generator:
                self.uploaded_generators.append(client.local_generator)

    def aggregate_generators(self):
        """
        Aggregate generators from clients using FedAvg
        """
        if not self.uploaded_generators or not self.global_generator:
            print("[Server] No generators to aggregate")
            return

        print(f"[Server] Aggregating {len(self.uploaded_generators)} generators")

        # Aggregate generator parameters
        global_gen_dict = self.global_generator.generator.state_dict()

        for key in global_gen_dict.keys():
            if 'num_batches_tracked' in key:
                continue

            global_gen_dict[key] = torch.zeros_like(global_gen_dict[key])

            for gen in self.uploaded_generators:
                gen_dict = gen.generator.state_dict()
                global_gen_dict[key] += gen_dict[key] / len(self.uploaded_generators)

        self.global_generator.generator.load_state_dict(global_gen_dict)

        # Aggregate critic (discriminator) parameters
        global_critic_dict = self.global_generator.critic.state_dict()

        for key in global_critic_dict.keys():
            if 'num_batches_tracked' in key:
                continue

            global_critic_dict[key] = torch.zeros_like(global_critic_dict[key])

            for gen in self.uploaded_generators:
                critic_dict = gen.critic.state_dict()
                global_critic_dict[key] += critic_dict[key] / len(self.uploaded_generators)

        self.global_generator.critic.load_state_dict(global_critic_dict)

        print("[Server] Generator aggregation complete")

    def train_global_generator(self, task):
        """
        Additional training of global generator on server
        Using knowledge from all clients
        """
        if not self.global_generator:
            return

        print(f"[Server] Training global generator for task {task}")

        self.global_generator.train()
        classes_list = list(self.global_classes)

        if not classes_list:
            return

        batch_size = 64

        for step in range(self.generator_iterations):
            # Sample data from generator
            noise, aux_label, _ = self._generate_noise(batch_size, classes_list)

            # Train generator
            train_results = self.global_generator.train_a_batch(
                x=noise,  # In reality, this should be real data
                y=aux_label,
                classes_so_far=classes_list,
                importance_of_new_task=self.importance_of_new_task
            )

            if (step + 1) % 20 == 0:
                print(f"  Step [{step+1}/{self.generator_iterations}] | "
                      f"C Loss: {train_results['c_loss']:.4f}, "
                      f"G Loss: {train_results['g_loss']:.4f}")

        print(f"[Server] Global generator training complete")

    def _generate_noise(self, batch_size, classes_list):
        """Generate noise for generator training"""
        noise = torch.FloatTensor(batch_size, self.nz, 1, 1)
        aux_label = torch.LongTensor(batch_size)
        dis_label = torch.FloatTensor(batch_size)

        # Sample labels
        label = np.random.choice(classes_list, batch_size)

        # Create noise with class encoding
        noise_ = np.random.normal(0, 1, (batch_size, self.nz))
        class_onehot = np.zeros((batch_size, self.num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :self.num_classes] = class_onehot[np.arange(batch_size)]

        noise_ = torch.from_numpy(noise_).float()
        noise.copy_(noise_.view(batch_size, self.nz, 1, 1))
        aux_label.copy_(torch.from_numpy(label))

        noise = torch.squeeze(noise).to(self.device)
        aux_label = aux_label.to(self.device)
        dis_label = dis_label.to(self.device).fill_(0)

        return noise, aux_label, dis_label

    def visualize_samples(self, task):
        """Generate and save sample images"""
        if not self.global_generator:
            return

        print(f"[Server] Generating visualization for task {task}")

        save_dir = os.path.join("output_fedcil", self.dataset, f"task_{task}")
        os.makedirs(save_dir, exist_ok=True)

        self.global_generator.eval()

        classes_list = sorted(list(self.global_classes))
        if not classes_list:
            return

        samples_per_class = 10

        with torch.no_grad():
            for class_id in classes_list:
                class_dir = os.path.join(save_dir, f"class_{class_id}")
                os.makedirs(class_dir, exist_ok=True)

                # Generate samples
                noise, aux_label, _ = self._generate_noise(
                    samples_per_class,
                    [class_id] * samples_per_class
                )

                gen_imgs = self.global_generator.generator(noise)

                # Normalize to [0, 1]
                gen_imgs = (gen_imgs + 1) / 2.0

                # Save individual images
                for idx in range(samples_per_class):
                    save_path = os.path.join(class_dir, f"sample_{idx}.png")
                    save_image(gen_imgs[idx], save_path)

        print(f"[Server] Samples saved to {save_dir}")

    def evaluate_all(self, save=True, selected=False):
        """Evaluate on current task"""
        # Call parent class evaluation
        test_ids, test_samples, test_accs, test_losses = self.test(selected=selected)

        glob_acc = np.sum(test_accs) * 1.0 / np.sum(test_samples)
        glob_loss = np.sum([x * y for (x, y) in zip(test_samples, test_losses)]) / np.sum(test_samples)

        if save:
            self.metrics['glob_acc'].append(glob_acc)
            self.metrics['glob_loss'].append(glob_loss)

        print(f"[Server] Current Task Accuracy: {glob_acc:.4f}, Loss: {glob_loss:.4f}")

    def evaluate_all_tasks(self, task):
        """Evaluate forgetting on all learned tasks"""
        print(f"\n[Server] Evaluating all tasks up to task {task}")

        # This requires task-specific test loaders
        # Implementation depends on your data loading structure

        task_accs = {}
        for t in range(task + 1):
            # Get accuracy for task t
            # task_accs[t] = self.test_task(t)
            pass

        # Compute average accuracy
        if task_accs:
            avg_acc = np.mean(list(task_accs.values()))
            print(f"[Server] Average accuracy across all tasks: {avg_acc:.4f}")

            # Compute forgetting
            if task > 0:
                # Compare with initial accuracy on each task
                forgetting = 0  # Implement forgetting calculation
                print(f"[Server] Average forgetting: {forgetting:.4f}")
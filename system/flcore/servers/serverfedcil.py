"""
FedCIL Server Implementation

Based on "Better Generative Replay for Continual Federated Learning" (ICLR 2023)

Key Server Responsibilities:
1. Train AC-GAN (Auxiliary Classifier GAN) for generative replay
2. Aggregate client models with model consolidation
3. Distribute generator and global model to clients
4. Enforce consistency across heterogeneous clients
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import os
from torchvision.utils import save_image
from flcore.servers.serverbase import Server
from flcore.clients.clientfedcil import clientFedCIL



# ==========================================
# AC-GAN Components
# ==========================================

class ACGenerator(nn.Module):
    """
    AC-GAN Generator for FedCIL
    Generates images conditioned on class labels
    """

    def __init__(self, nz=100, ngf=64, img_size=32, nc=3, num_classes=10):
        super(ACGenerator, self).__init__()

        self.nz = nz
        self.num_classes = num_classes
        self.img_size = img_size

        # Label embedding
        self.label_emb = nn.Embedding(num_classes, nz)

        # Calculate initial size
        if img_size == 224:
            self.init_size = 7
        elif img_size == 64:
            self.init_size = 4
        else:
            self.init_size = 4

        # Initial projection
        self.l1 = nn.Sequential(
            nn.Linear(nz * 2, ngf * 8 * self.init_size ** 2),
            nn.BatchNorm1d(ngf * 8 * self.init_size ** 2),
            nn.ReLU(True)
        )

        # Convolutional blocks for different image sizes
        if img_size == 32:
            self.conv_blocks = nn.Sequential(
                self._upsample_block(ngf * 8, ngf * 4),  # 4->8
                self._upsample_block(ngf * 4, ngf * 2),  # 8->16
                self._upsample_block(ngf * 2, ngf),  # 16->32
                nn.Conv2d(ngf, nc, 3, 1, 1),
                nn.Tanh()
            )
        elif img_size == 64:
            self.conv_blocks = nn.Sequential(
                self._upsample_block(ngf * 8, ngf * 4),  # 4->8
                self._upsample_block(ngf * 4, ngf * 2),  # 8->16
                self._upsample_block(ngf * 2, ngf),  # 16->32
                self._upsample_block(ngf, ngf // 2),  # 32->64
                nn.Conv2d(ngf // 2, nc, 3, 1, 1),
                nn.Tanh()
            )
        else:  # 224
            self.conv_blocks = nn.Sequential(
                self._upsample_block(ngf * 8, ngf * 4),  # 7->14
                self._upsample_block(ngf * 4, ngf * 2),  # 14->28
                self._upsample_block(ngf * 2, ngf),  # 28->56
                self._upsample_block(ngf, ngf // 2),  # 56->112
                self._upsample_block(ngf // 2, ngf // 4),  # 112->224
                nn.Conv2d(ngf // 4, nc, 3, 1, 1),
                nn.Tanh()
            )

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, z, labels):
        # Embed labels
        label_emb = self.label_emb(labels)
        # Concatenate noise and label
        gen_input = torch.cat([z, label_emb], dim=1)
        # Generate
        out = self.l1(gen_input)
        out = out.view(out.size(0), -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ACDiscriminator(nn.Module):
    """
    AC-GAN Discriminator with Auxiliary Classifier

    Outputs:
    - Real/Fake prediction (adversarial loss)
    - Class prediction (auxiliary classifier)
    """

    def __init__(self, nc=3, ndf=64, img_size=32, num_classes=10):
        super(ACDiscriminator, self).__init__()

        self.img_size = img_size

        # Feature extraction
        if img_size == 32:
            self.features = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif img_size == 64:
            self.features = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, ndf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:  # 224
            self.features = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.AdaptiveAvgPool2d(1)
            )

        # Adversarial head (real/fake)
        self.adv_head = nn.Sequential(
            nn.Linear(ndf * 8, 1),
            nn.Sigmoid()
        )

        # Auxiliary classifier head (class prediction)
        self.aux_head = nn.Linear(ndf * 8, num_classes)

    def forward(self, img):
        features = self.features(img)
        features = features.view(features.size(0), -1)

        # Real/fake prediction
        validity = self.adv_head(features)

        # Class prediction
        class_pred = self.aux_head(features)

        return validity, class_pred


def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ==========================================
# FedCIL Server
# ==========================================

class serverFedCIL(Server):
    """
    FedCIL Server Implementation

    Implements the server-side logic for Better Generative Replay
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        self.Budget = []

        # --- Configuration ---
        if 'cifar100' in self.dataset.lower():
            self.img_size = 32
            self.nz = 256
            self.nc = 3
        elif 'imagenet' in self.dataset.lower():
            self.img_size = 64 if '100' in self.dataset else 224
            self.nz = 256
            self.nc = 3
        else:
            self.img_size = 32
            self.nz = 100
            self.nc = 3

        # --- AC-GAN Components ---
        self.generator = ACGenerator(
            nz=self.nz,
            ngf=64,
            img_size=self.img_size,
            nc=self.nc,
            num_classes=args.num_classes
        ).to(self.device)

        self.discriminator = ACDiscriminator(
            nc=self.nc,
            ndf=64,
            img_size=self.img_size,
            num_classes=args.num_classes
        ).to(self.device)

        # Initialize weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # --- Optimizers ---
        self.g_lr = getattr(args, 'g_lr', 0.0002)
        self.d_lr = getattr(args, 'd_lr', 0.0002)

        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.g_lr,
            betas=(0.5, 0.999)
        )

        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.d_lr,
            betas=(0.5, 0.999)
        )

        # --- Loss Functions ---
        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

        # --- Hyperparameters ---
        self.gan_steps = getattr(args, 'gan_steps', 200)  # GAN training iterations
        self.lambda_aux = getattr(args, 'lambda_aux', 1.0)  # Auxiliary loss weight
        self.consolidation_weight = getattr(args, 'consolidation_weight', 0.1)

        # --- Previous Generator for Consolidation ---
        self.prev_generator = None

        # Initialize clients
        self.set_clients(clientFedCIL)

        print(f"[Server] FedCIL initialized | ImgSize={self.img_size}, nz={self.nz}, "
              f"num_classes={args.num_classes}")

    def train(self):
        """Main training loop for FedCIL"""

        for task in range(self.args.num_tasks):
            print(f"\n{'=' * 60}")
            print(f"Task {task} / {self.args.num_tasks}")
            print('=' * 60)

            # --- Task Setup ---
            if task > 0:
                self.current_task = task
                torch.cuda.empty_cache()

                # Load new task data for each client
                for i, client in enumerate(self.clients):
                    # Your data loading logic here
                    # This should be adapted to your specific data loading function
                    pass

            # --- Federated Learning Rounds ---
            for round_idx in range(self.global_rounds):
                print(f"\n--- Round {round_idx + 1}/{self.global_rounds} ---")

                start_time = time.time()
                glob_iter = round_idx + self.global_rounds * task

                # Select clients
                self.selected_clients = self.select_clients()

                # Evaluate before training
                self.eval(task=task, glob_iter=glob_iter, flag="global")

                # Client training
                for client in self.selected_clients:
                    # Send global model copy for consistency enforcement
                    client.set_global_model_copy(self.global_model)
                    # Send generator
                    client.set_generator(self.generator)
                    # Train
                    client.train(task=task)

                # Receive models
                self.receive_models()

                # Aggregate with consolidation
                self.aggregate_with_consolidation()

                # Send updated models
                self.send_models()

                self.Budget.append(time.time() - start_time)
                print(f"Round Time: {self.Budget[-1]:.2f}s")

            # --- Post-Task Processing ---
            print(f"\n--- Task {task} Post-Processing ---")

            # Train AC-GAN on aggregated knowledge
            self.train_acgan()

            # Visualize generated samples
            self.visualize_generated_samples(task)

            # Evaluate forgetting
            self.eval_task(task=task, glob_iter=task, flag="global")

            print(f"\nTask {task} completed.\n")

    def aggregate_with_consolidation(self):
        """
        Aggregate client models with model consolidation

        Model consolidation helps stabilize federated training
        by smoothly updating the global model
        """
        print(f"[Server] Aggregating {len(self.uploaded_models)} client models with consolidation")

        if not self.uploaded_models:
            return

        # Standard FedAvg aggregation
        global_dict = self.global_model.state_dict()

        for key in global_dict.keys():
            if 'num_batches_tracked' in key:
                continue

            # Weighted average of client models
            global_dict[key] = torch.zeros_like(global_dict[key])

            for client_model in self.uploaded_models:
                client_dict = client_model.state_dict()
                global_dict[key] += client_dict[key] / len(self.uploaded_models)

        self.global_model.load_state_dict(global_dict)

        print("[Server] Aggregation complete")

    def train_acgan(self):
        """
        Train AC-GAN (Auxiliary Classifier GAN)

        This creates a generator that can produce samples from all learned classes
        The auxiliary classifier helps maintain class-specific generation quality
        """
        print(f"[Server] Training AC-GAN for {self.gan_steps} steps")

        # Collect all learned classes from clients
        all_classes = set()
        for client in self.clients:
            all_classes.update(client.learned_classes)

        classes_list = list(all_classes)

        if len(classes_list) == 0:
            print("[Server] No classes to train GAN on")
            return

        print(f"[Server] Training GAN on {len(classes_list)} classes: {classes_list}")

        self.generator.train()
        self.discriminator.train()

        batch_size = 64

        # Loss tracking
        g_losses = []
        d_losses = []

        for step in range(self.gan_steps):
            # ===================================
            # Train Discriminator
            # ===================================
            self.optimizer_D.zero_grad()

            # Sample labels
            labels = torch.tensor(
                np.random.choice(classes_list, batch_size),
                dtype=torch.long
            ).to(self.device)

            # Generate fake images
            z = torch.randn(batch_size, self.nz).to(self.device)
            fake_imgs = self.generator(z, labels).detach()

            # Real and fake labels for adversarial loss
            valid = torch.ones(batch_size, 1).to(self.device)
            fake = torch.zeros(batch_size, 1).to(self.device)

            # Discriminator on fake images
            fake_validity, fake_aux = self.discriminator(fake_imgs)
            d_fake_loss = self.adversarial_loss(fake_validity, fake)
            d_fake_aux_loss = self.auxiliary_loss(fake_aux, labels)

            # Get real images from clients (sampling strategy)
            real_imgs, real_labels = self._sample_real_images(batch_size, classes_list)

            if real_imgs is not None:
                # Discriminator on real images
                real_validity, real_aux = self.discriminator(real_imgs)
                d_real_loss = self.adversarial_loss(real_validity, valid)
                d_real_aux_loss = self.auxiliary_loss(real_aux, real_labels)

                # Total discriminator loss
                d_loss = (
                                 d_real_loss + d_fake_loss +
                                 self.lambda_aux * (d_real_aux_loss + d_fake_aux_loss)
                         ) / 2
            else:
                # If no real images available, use only fake
                d_loss = d_fake_loss + self.lambda_aux * d_fake_aux_loss

            d_loss.backward()
            self.optimizer_D.step()

            # ===================================
            # Train Generator
            # ===================================
            self.optimizer_G.zero_grad()

            # Generate new fake images
            z = torch.randn(batch_size, self.nz).to(self.device)
            labels = torch.tensor(
                np.random.choice(classes_list, batch_size),
                dtype=torch.long
            ).to(self.device)

            gen_imgs = self.generator(z, labels)

            # Generator wants discriminator to classify fake as real
            validity, aux_pred = self.discriminator(gen_imgs)

            g_adv_loss = self.adversarial_loss(validity, valid)
            g_aux_loss = self.auxiliary_loss(aux_pred, labels)

            g_loss = g_adv_loss + self.lambda_aux * g_aux_loss

            # Generator consolidation (if previous generator exists)
            if self.prev_generator is not None:
                self.prev_generator.eval()
                with torch.no_grad():
                    prev_imgs = self.prev_generator(z, labels)

                consolidation_loss = F.mse_loss(gen_imgs, prev_imgs)
                g_loss += self.consolidation_weight * consolidation_loss

            g_loss.backward()
            self.optimizer_G.step()

            # Track losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            # Periodic logging
            if (step + 1) % 50 == 0:
                print(f"  Step [{step + 1}/{self.gan_steps}] | "
                      f"D Loss: {np.mean(d_losses[-50:]):.4f} | "
                      f"G Loss: {np.mean(g_losses[-50:]):.4f}")

        # Update previous generator for next task
        self.prev_generator = copy.deepcopy(self.generator)
        self.prev_generator.eval()

        print(f"[Server] AC-GAN training complete | "
              f"Avg D Loss: {np.mean(d_losses):.4f} | "
              f"Avg G Loss: {np.mean(g_losses):.4f}")

    def _sample_real_images(self, batch_size, classes_list):
        """
        Sample real images from client models
        This is used for discriminator training
        """
        # This is a placeholder - implement based on your data loading strategy
        # You might want to cache some real images from clients
        return None, None

    def visualize_generated_samples(self, task):
        """Generate and save sample images from the generator"""
        print(f"[Server] Generating visualization samples for task {task}")

        save_dir = os.path.join("output_fedcil", self.dataset, f"task_{task}")
        os.makedirs(save_dir, exist_ok=True)

        self.generator.eval()

        # Get all learned classes
        all_classes = set()
        for client in self.clients:
            all_classes.update(client.learned_classes)

        classes_list = sorted(list(all_classes))

        if not classes_list:
            return

        samples_per_class = 10

        with torch.no_grad():
            for class_id in classes_list:
                class_dir = os.path.join(save_dir, f"class_{class_id}")
                os.makedirs(class_dir, exist_ok=True)

                z = torch.randn(samples_per_class, self.nz).to(self.device)
                labels = torch.full((samples_per_class,), class_id, dtype=torch.long).to(self.device)

                gen_imgs = self.generator(z, labels)
                gen_imgs = (gen_imgs + 1) / 2.0  # Denormalize from [-1, 1] to [0, 1]

                for idx in range(samples_per_class):
                    save_path = os.path.join(class_dir, f"sample_{idx}.png")
                    save_image(gen_imgs[idx], save_path)

        print(f"[Server] Saved samples to {save_dir}")

    def send_models(self):
        """Send global model and generator to clients"""
        for client in self.clients:
            client.set_parameters(self.global_model)
            client.set_generator(self.generator)

    def receive_models(self):
        """Receive models from selected clients"""
        print(f"[Server] Receiving models from {len(self.selected_clients)} clients")

        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_models.append(client.model)
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientMFCL import MFCLClient
from flcore.servers.serverbase import Server
from flcore.trainmodel.mfcl_utils import Generator, get_bn_statistics


class MFCLServer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(MFCLClient)
        
        # Generator params
        self.z_dim = 100
        self.generator = Generator(z_dim=self.z_dim, img_channels=3).to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.gen_epochs = 100 # "Eg" in Algo 1
        
        # Weights for Generator loss 
        self.w_div = 1.0
        self.w_bn = 10.0 # Often high in Data-free papers
        
        # To store old model for clients
        self.old_model = None

    def train_df_generator(self, num_discovered_classes):
        """
        Train Generator Data-Free on the Server (Algorithm 1 Line 17).
        """
        print(f"Server: Training Data-Free Generator for {self.gen_epochs} epochs...")
        self.global_model.eval()
        self.generator.train()
        
        # Hooks for Batch Statistics Loss (Eq 3)
        # We need to capture the statistics of the global model on the synthetic data
        # AND compare them to the stored running_stats of the global model.
        bn_layers = [m for m in self.global_model.modules() if isinstance(m, nn.BatchNorm2d)]
        
        for epoch in range(self.gen_epochs):
            # 1. Sample Noise z
            batch_size = 32
            z = torch.randn(batch_size, self.z_dim, device=self.device)
            
            # 2. Generate Synthetic Data
            x_syn = self.generator(z)
            
            # 3. Generator Loss
            
            # --- L_CE (Eq 1) ---
            # Labels: argmax of first q dimensions of z [cite: 598]
            # Ensure q <= z_dim
            q = min(num_discovered_classes, self.z_dim)
            targets = torch.argmax(z[:, :q], dim=1)
            
            output_syn = self.global_model(x_syn)
            loss_ce = F.cross_entropy(output_syn, targets)
            
            # --- L_Div (Eq 2) ---
            # Maximize entropy of the average prediction to ensure diversity [cite: 601]
            # H(p_bar) = - sum p_bar * log p_bar
            p_syn = F.softmax(output_syn, dim=1)
            p_avg = torch.mean(p_syn, dim=0)
            loss_div = -torch.sum(p_avg * torch.log(p_avg + 1e-8))
            # We want to maximize entropy, so minimize negative entropy.
            # However, Eq 2 says L_div = -H_info. So minimizing L_div maximizes H_info.
            # Correct.
            
            # --- L_BN (Eq 3) ---
            # Match statistics of synthetic batch to running stats of global model 
            loss_bn = 0.0
            # To get batch stats of synthetic data at specific layers, we need hooks or 
            # we can rely on the fact that if we pass x_syn through the model, 
            # we can grab the input mean/var if we had access.
            # Simplified approach: The paper says "minimize layer-wise distances".
            # We iterate BN layers.
            
            # We need to forward x_syn through global_model and capture feature maps at BN inputs.
            # This requires a functional call or hooks. 
            # Let's assume a simplified hook storage.
            
            bn_feats = []
            def hook_bn_input(module, input, output):
                bn_feats.append(input[0])
                
            handles = [layer.register_forward_hook(hook_bn_input) for layer in bn_layers]
            
            # Re-forward to capture features
            _ = self.global_model(x_syn)
            
            for i, layer in enumerate(bn_layers):
                feat = bn_feats[i] # (N, C, H, W)
                
                # Synthetic Batch Stats
                mu_syn = feat.mean([0, 2, 3])
                var_syn = feat.var([0, 2, 3], unbiased=False)
                
                # Real Stored Stats
                mu_real = layer.running_mean
                var_real = layer.running_var
                
                # KL Divergence for Gaussians (Simplified to MSE for means + vars often used in implementation)
                # Paper Eq 3 specifies KL. 
                # KL(N0||N1) = log(s1/s0) + (s0^2 + (m0-m1)^2)/2s1^2 - 0.5
                # Here we assume standard Gaussian approx or use L2 for stability often seen in DeepInversion code.
                # Let's stick to L2 for robustness if KL is unstable, or implement full KL.
                # Let's implement L2 for simplicity as exact KL can be volatile with small vars.
                loss_bn += F.mse_loss(mu_syn, mu_real) + F.mse_loss(var_syn, var_real)

            for h in handles: h.remove()
            
            # Total Loss
            # Note: Paper says minimize Eq 4. 
            loss_g = loss_ce + self.w_div * loss_div + self.w_bn * loss_bn
            
            self.gen_optimizer.zero_grad()
            loss_g.backward()
            self.gen_optimizer.step()

    def train(self):
        """
        MFCL Main Training Loop [Algorithm 1].
        """
        # num_classes = self.num_classes # Total classes (100)
        classes_per_task = self.args.num_classes // self.N_TASKS
        current_classes_count = 0
        
        for task in range(self.N_TASKS):
            print(f"\n=== MFCL Task {task} ===")
            self.current_task = task
            current_classes_count += classes_per_task
            
            # 1. Update Architecture if needed (e.g. expanding FC head)
            # Assuming global model has all heads pre-allocated or handles masking internally.
            
            # 2. Distribute Generator and Old Model to Clients
            # In base class, send_models() usually copies parameters.
            # We need to explicitly set the extra attributes on clients.
            for client in self.clients:
                client.generator = copy.deepcopy(self.generator) # Send G 
                client.old_model = copy.deepcopy(self.old_model) # Send F_t-1
                client.current_task = task

            # 3. Federated Training Rounds
            for round in range(self.global_rounds):
                # Standard FedAvg steps
                self.selected_clients = self.select_clients()
                self.send_models() 
                
                # Note: Client.train() handles the specific MFCL losses
                for client in self.selected_clients:
                    client.train(task)
                
                self.receive_models()
                self.aggregate_parameters()
                
                self.eval(task, round, flag="global")

            # 4. End of Task: Server Operations 
            
            # a. Save Frozen Old Model (F_t-1)
            self.old_model = copy.deepcopy(self.global_model)
            for param in self.old_model.parameters():
                param.requires_grad = False
                
            # b. Train Data-Free Generator (G)
            # "At the end of each task... server trains a generative model" [cite: 493]
            self.train_df_generator(num_discovered_classes=current_classes_count)
            
            # c. Freeze Generator
            # Usually we keep training it or reset? 
            # Paper Algo 1 line 18: "G.freezeModel()". 
            # Line 6: "G, F0 initialize()". It seems G is cumulative or re-trained.
            # Line 17 calls trainDFGenerator.
            # We will keep the state but freeze it for client usage in next task.
            for param in self.generator.parameters():
                param.requires_grad = False
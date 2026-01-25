import copy

import torch
import torch.nn as nn
import torch.nn.functional as F  # Cần thêm cái này nếu chưa có
from flcore.clients.clientbase import Client


class MFCLClient(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        self.generator = None
        self.old_model = None
        self.z_dim = 100
        self.w_ft = 1.0 
        self.w_kd = 1.0

    def train(self, task):
        """
        MFCL Client Training Loop.
        """
        trainloader = self.load_train_data(task=task)
        self.model.to(self.device)
        self.model.train()
        
        # Prepare old model for Distillation if not first task
        if self.current_task > 0 and self.old_model is not None:
            self.old_model.to(self.device)
            self.old_model.eval()
            self.generator.to(self.device)
            self.generator.eval()

        # --- PHẦN SỬA LỖI (HOOK) ---
        self.features_curr = None
        self.features_old = None
        
        def hook_curr(module, input, output):
            # output của self.model.base chính là features (đã qua avgpool và identity)
            self.features_curr = output.flatten(1)
            
        def hook_old(module, input, output):
            self.features_old = output.flatten(1)
            
        # Đăng ký hook vào 'base' thay vì 'avgpool'
        # Vì model là BaseHeadSplit, nên self.model.base là backbone
        handle_curr = self.model.base.register_forward_hook(hook_curr)
        
        if self.old_model:
            handle_old = self.old_model.base.register_forward_hook(hook_old)
        # ---------------------------

        for epoch in range(self.local_epochs):
            for x_real, y_real in trainloader:
                x_real, y_real = x_real.to(self.device), y_real.to(self.device)
                bs = x_real.shape[0]

                self.optimizer.zero_grad()
                
                # --- 1. Current Task Loss (L_CE) ---
                out_real = self.model(x_real)
                loss_ce = self.loss(out_real, y_real)

                loss_ft = torch.tensor(0., device=self.device)
                loss_kd = torch.tensor(0., device=self.device)

                if self.current_task > 0 and self.generator is not None:
                    # --- Generate Synthetic Data ---
                    z = torch.randn(bs, self.z_dim, device=self.device)
                    with torch.no_grad():
                        x_syn = self.generator(z)
                        
                        # Tạo nhãn giả từ z (theo logic paper hoặc server training)
                        # Giả định server train với q class đầu tiên
                        q_prev = self.args.num_classes // self.args.nt * self.current_task 
                        y_syn = torch.argmax(z[:, :q_prev], dim=1)
                    
                    # --- 2. Previous Task Loss (L_FT) ---
                    # Forward synthetic data
                    out_syn = self.model(x_syn)
                    loss_ft = self.loss(out_syn, y_syn)

                    # --- 3. Knowledge Distillation (L_KD) ---
                    # Combine batch để trigger hook một lần (tiết kiệm forward pass)
                    # Tuy nhiên code ở trên đã forward riêng lẻ, nên features_curr 
                    # hiện tại đang chứa features của x_syn (từ lệnh out_syn = ...).
                    
                    # Để đúng logic Eq 7: || W(F_curr(x_syn)) - W(F_old(x_syn)) ||
                    # Ta cần features của x_syn từ cả model mới và model cũ.
                    
                    # Lấy features hiện tại của x_syn (đã có từ hook sau lệnh out_syn = ...)
                    feats_curr_syn = self.features_curr
                    
                    with torch.no_grad():
                        _ = self.old_model(x_syn) # Trigger hook features_old
                        feats_old_syn = self.features_old
                        
                    # Chiếu qua head cũ (W)
                    # Trong BaseHeadSplit, head nằm ở self.old_model.head
                    proj_curr = self.old_model.head(feats_curr_syn)
                    proj_old = self.old_model.head(feats_old_syn)
                    
                    loss_kd = torch.norm(proj_curr - proj_old, p=2) ** 2

                # Total Loss
                loss = loss_ce + self.w_ft * loss_ft + self.w_kd * loss_kd
                
                loss.backward()
                self.optimizer.step()

        # Remove hooks
        handle_curr.remove()
        if self.current_task > 0:
            handle_old.remove()
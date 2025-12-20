import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy
from flcore.clients.clientbase import Client
from flcore.utils_core.target_utils import Generator
import torch
import torch.nn as nn

def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

class clientOursV2(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        # Cấu hình Generator
        if 'cifar' in args.dataset.lower():
            self.img_size = 32; self.nz = 512
        elif 'imagenet' in args.dataset.lower():
            self.img_size = 64 if '100' in args.dataset else 224; self.nz = 256
        else:
            self.img_size = 64; self.nz = 100
            
        self.nc = 3
        # Trọng số cho Loss Replay và nhiệt độ T cho KD
        self.replay_weight = args.replay_weight if hasattr(args, 'replay_weight') else 1.0
        self.T = args.T if hasattr(args, 'T') else 2.0 

        # Khởi tạo Generator
        self.generator = Generator(
            nz=self.nz, ngf=64, img_size=self.img_size, nc=self.nc, device=self.device,
            num_classes=args.num_classes,
        ).to(self.device)
        self.freeze_generator()

        self.old_network = None # Teacher model

    def freeze_generator(self):
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

    def set_generator_parameters(self, global_generator):
        self.generator.load_state_dict(global_generator.state_dict())
        self.freeze_generator()

    def next_task(self, train, label_info=None, if_label=True):
        """Override để lưu lại model cũ làm Teacher trước khi học task mới"""
        super().next_task(train, label_info, if_label)
        
        # Snapshot model hiện tại thành old_network (Teacher)
        self.old_network = copy.deepcopy(self.model)
        self.old_network.eval()
        for param in self.old_network.parameters():
            param.requires_grad = False

    def train(self, task=None):
        trainloader = self.load_train_data(task=task)
        print(f"[Client-side] ID: {self.id}, task-id: {task}")
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()
        self.generator.eval()
        
        # 
        # Định nghĩa KL Loss function
        current_criterion = nn.CrossEntropyLoss()

        start_time = time.time()

        for epoch in range(self.local_epochs):
            for i, (x_real, y_real) in enumerate(trainloader):
                self.unique_labels.update(y_real.cpu().tolist())
                if isinstance(x_real, list): x_real = x_real[0]
                x_real = x_real.to(self.device)
                y_real = y_real.to(self.device)

                # --- 1. Real Data: CrossEntropy Loss ---
                output_real = self.model(x_real)
                loss_real = current_criterion(output_real, y_real)

                # --- 2. Replay Data: KL Divergence Loss (Distillation) ---
                loss_replay = 0
                # Chỉ Replay khi có Teacher (old_network) và task > 0
                if self.old_network is not None:
                    batch_size = x_real.shape[0]
                    
                    # a. Sinh dữ liệu giả
                    z = torch.randn(batch_size, self.nz).to(self.device)
                    # Chọn nhãn từ các task CŨ để ôn tập
                    # Lưu ý: Teacher chỉ biết kiến thức cũ
                    if len(self.classes_past_task) > 0:
                        fake_labels = np.random.choice(self.classes_past_task, batch_size)
                        fake_labels = torch.tensor(fake_labels).long().to(self.device)
                        
                        with torch.no_grad():
                            x_fake = self.generator(z, fake_labels)
                            # Teacher (Old Model) dự đoán soft labels
                            teacher_logits = self.old_network(x_fake)

                        # Student (Current Model) dự đoán
                        student_logits = self.model(x_fake.detach())

                        # b. Tính KL Loss: Student cố gắng khớp distribution của Teacher
                        # Công thức: KL(log_softmax(S/T), softmax(T/T)) * T^2
                        loss_kd = _KD_loss(
                            pred=student_logits,
                            soft=teacher_logits,
                            T=self.T
                        )
                        loss_ce = F.cross_entropy(
                            student_logits,fake_labels
                        )
                        
                        loss_replay = loss_ce + self.args.kd * loss_kd

                # --- 3. Tổng hợp Loss ---
                total_loss = loss_real + self.replay_weight * loss_replay
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
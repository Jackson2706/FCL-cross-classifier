import copy
import torch
import torch.nn as nn
import numpy as np
import statistics
from torch.utils.data import DataLoader, ConcatDataset
from utils.data_utils import *

from flcore.trainmodel.models import *

import os


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_data, task_sequence, **kwargs):
        torch.manual_seed(0)
        self.t_angle_after = 0

        self.model = copy.deepcopy(args.model)
        self.args = args
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  

        self.num_classes = args.num_classes
        self.train_data = train_data
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        self.train_source = [image for image, _ in self.train_data]
        self.train_targets = [label for _, label in self.train_data]

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()

        if args.algorithm != "PreciseFCL":
            if args.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            elif args.optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {args.optimizer}.")
            self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, 
                gamma=args.learning_rate_decay_gamma
            )
            self.learning_rate_decay = args.learning_rate_decay

        self.classes_so_far = [] # all labels of a client so far 
        self.available_labels_current = [] # labels from all clients on T (current)
        self.current_labels = [] # current labels for itself
        self.classes_past_task = [] # classes_so_far (current labels excluded) 
        self.available_labels_past = [] # labels from all clients on T-1
        self.available_labels = [] # l from all c from 0-T
        self.current_task = 0
        self.task_dict = {}
        self.last_copy = None
        self.if_last_copy = False
        self.args = args
        self.task_sequence = task_sequence
        self.curriculum_step = 0
        self.current_task_id = task_sequence[0]
        self.tasks_seen_so_far = []
        self.unique_labels = set()

    def next_task(self, train, label_info = None, if_label = True):
        
        if self.args.algorithm != "PreciseFCL" and self.learning_rate_decay:
            # update last model:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate  # Đặt lại về giá trị ban đầu

            self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer, 
                gamma=self.args.learning_rate_decay_gamma
            )

        self.last_copy  = copy.deepcopy(self.model)
        self.last_copy.cuda()
        self.if_last_copy = True
        
        # update dataset: 
        self.train_data = train
        self.train_targets = [label for _, label in self.train_data]
        
        self.classes_past_task = copy.deepcopy(self.classes_so_far)
        self.current_task += 1

        # update classes_so_far
        if if_label:
            self.classes_so_far.extend(label_info['labels'])
            self.task_dict[self.current_task] = label_info['labels']

            self.current_labels.clear()
            self.current_labels.extend(label_info['labels'])
            
        return

    def assign_task_id(self, task_dict):
        if not isinstance(task_dict, dict):
            raise ValueError("task_dict must be a dictionary")

        label_key = tuple(sorted(self.current_labels)) if isinstance(self.current_labels,
                                                                     (set, list)) else self.current_labels

        return task_dict.get(label_key, -1)  # Returns -1 if labels are not in task_dict

    def load_train_data(self, task, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        if self.args.dataset == 'IMAGENET1k':
            train_data = read_client_data_FCL_imagenet1k(self.id, task=task, classes_per_task=self.args.cpt, count_labels=False, train=True)
        elif self.args.dataset == 'CIFAR100':
            train_data = read_client_data_FCL_cifar100(self.id, task=task, classes_per_task=self.args.cpt, count_labels=False, train=True)
        elif self.args.dataset == 'CIFAR10':
            train_data = read_client_data_FCL_cifar10(self.id, task=task, classes_per_task=self.args.cpt, count_labels=False, train=True)
        
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, task, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        if self.args.dataset == 'IMAGENET1k':
            test_data = read_client_data_FCL_imagenet1k(self.id, task=task, classes_per_task=self.args.cpt, count_labels=False, train=False)
        elif self.args.dataset == 'CIFAR100':
            test_data = read_client_data_FCL_cifar100(self.id, task=task, classes_per_task=self.args.cpt, count_labels=False, train=False)
        elif self.args.dataset == 'CIFAR10':
            test_data = read_client_data_FCL_cifar10(self.id, task=task, classes_per_task=self.args.cpt, count_labels=False, train=False)

        return test_data  

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self, task, model_to_use=None):
        # Nếu không truyền model, mặc định dùng model của client
        model = model_to_use if model_to_use is not None else self.model
        model.eval()
        
        test_data = self.load_test_data(task=task)
        loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x)
                correct += (torch.argmax(out, dim=1) == y).sum().item()
                total += y.size(0)
        return correct, total

    def train_metrics(self, task):
        trainloader = self.load_train_data(task=task)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def grad_eval(self, old_model):
        # TODO Re-eval again
        network_test = []

        network_inner = []
        optimizer_inner = []
        optimizer_proto_inner = []
        optimizer_head_inner = []

        for task_id, task in enumerate(self.task_dict):
            temp_model = FedAvgCNN(in_features=3, num_classes=self.num_classes, dim=1600).to(self.device)
            temp_head = copy.deepcopy(temp_model.fc)
            temp_model.fc = nn.Identity()
            temp_model = BaseHeadSplit(temp_model, temp_head)

            temp_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            network_inner.append(temp_model)

            if self.args.optimizer == "sgd":
                optimizer_inner.append(
                    torch.optim.SGD(network_inner[task_id].parameters(), lr=self.learning_rate)
                )
                optimizer_proto_inner.append(
                    torch.optim.SGD(network_inner[task_id].base.parameters(), lr=self.learning_rate)
                )
                optimizer_head_inner.append(
                    torch.optim.SGD(network_inner[task_id].head.parameters(), lr=self.learning_rate)
                )
            elif self.args.optimizer == "adam":
                optimizer_inner.append(
                    torch.optim.Adam(network_inner[task_id].parameters(), lr=self.learning_rate)
                )
                optimizer_proto_inner.append(
                    torch.optim.Adam(network_inner[task_id].base.parameters(), lr=self.learning_rate)
                )
                optimizer_head_inner.append(
                    torch.optim.Adam(network_inner[task_id].head.parameters(), lr=self.learning_rate)
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.args.optimizer}.")

            optimizer_inner[task_id].load_state_dict(self.optimizer.state_dict())
            if self.args.coreset:
                optimizer_proto_inner[task_id].load_state_dict(self.optimizer_proto.state_dict())
                optimizer_head_inner[task_id].load_state_dict(self.optimizer_head.state_dict())

        for task_id, task in enumerate(self.task_dict):
            if self.args.coreset:
                trainloader = self.load_train_data(task=task)
                for epoch in range(self.local_epochs):
                    for i, (x, y) in enumerate(trainloader):
                        if type(x) == type([]):
                            x[0] = x[0].to(self.device)
                        else:
                            x = x.to(self.device)
                        y = y.to(self.device)

                        # TODO First Step: ProtoNet update
                        proto, output = network_inner[task_id].get_proto(x)
                        proto_metric = self.proto_loss(proto, y)
                        optimizer_proto_inner[task_id].zero_grad()
                        proto_metric[0].backward()
                        optimizer_proto_inner[task_id].step()

                        # TODO Second Step: Entire model update (Or classifier only?)
                        output = network_inner[task_id](x)
                        loss = self.loss(output, y)
                        optimizer_head_inner[task_id].zero_grad()
                        loss.backward()
                        optimizer_head_inner[task_id].step()

            else:  # TODO Base+Head use the same classification loss
                trainloader = self.load_train_data(task=task)
                for epoch in range(self.local_epochs):
                    for i, (x, y) in enumerate(trainloader):
                        if type(x) == type([]):
                            x[0] = x[0].to(self.device)
                        else:
                            x = x.to(self.device)
                        y = y.to(self.device)
                        # TODO Base+Head use the same classification loss
                        output = network_inner[task_id](x)
                        loss = self.loss(output, y)
                        optimizer_inner[task_id].zero_grad()
                        loss.backward()
                        optimizer_inner[task_id].step()

            network_test.append(network_inner[task_id])

        # TODO Measure Gradient Angles After Aggregate
        angle_value = []

        # for model_i in network_test:
        #     for model_j in network_test:
        #         angle_value.append(self.cos_sim(old_model, model_i, model_j))

        for i in range(len(network_test)):
            for j in range(i + 1, len(network_test)):
                angle_value.append(self.cos_sim(old_model, network_test[i], network_test[j]))

        if angle_value:
            self.t_angle_after = statistics.mean(angle_value)
        else:
            self.t_angle_after = 1
        print(f"AFTER  t angle:{self.t_angle_after}")

    def cos_sim(self, prev_model, model1, model2):
        prev_param = torch.cat([p.data.view(-1) for p in prev_model.parameters()])
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])

        grad1 = params1 - prev_param
        grad2 = params2 - prev_param

        cos_sim = torch.dot(grad1, grad2) / (torch.norm(grad1) * torch.norm(grad2))
        return cos_sim.item()

    def cosine_similarity(self, model1, model2):
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
        cos_sim = torch.dot(params1, params2) / (torch.norm(params1) * torch.norm(params2))
        return cos_sim.item()

    def distance(self, model1, model2):
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])

        mse = F.mse_loss(params1, params2)
        return mse.item()

    def move_to_next_task(self):
        # 1. Lưu task hiện tại vào lịch sử trước khi nhảy sang task mới
        if self.current_task_id not in self.tasks_seen_so_far:
            self.tasks_seen_so_far.append(self.current_task_id)

        # 2. Chuyển sang bước tiếp theo trong lộ trình (Temporal Heterogeneity)
        self.curriculum_step += 1
        
        if self.curriculum_step < len(self.task_sequence):
            # Lấy Task ID tiếp theo từ trình tự riêng của client này
            next_id = self.task_sequence[self.curriculum_step]
            self.current_task_id = next_id

            # 3. Tận dụng logic nạp dữ liệu (giống trong load_train_data của bạn)
            # Ở đây ta cần lấy 'train_data' gốc và 'label_info'
            if self.args.dataset == 'IMAGENET1k':
                train_data, label_info = read_client_data_FCL_imagenet1k(self.id, task=next_id, classes_per_task=self.args.cpt, count_labels=True, train=True)
            elif self.args.dataset == 'CIFAR100':
                train_data, label_info = read_client_data_FCL_cifar100(self.id, task=next_id, classes_per_task=self.args.cpt, count_labels=True, train=True)
            elif self.args.dataset == 'CIFAR10':
                train_data, label_info = read_client_data_FCL_cifar10(self.id, task=next_id, classes_per_task=self.args.cpt, count_labels=True, train=True)

            # 4. Gọi hàm next_task có sẵn của bạn để đồng bộ mô hình và nhãn
            # Hàm này của bạn sẽ cập nhật self.train_data, self.model_copy, v.v.
            self.next_task(train=train_data, label_info=label_info, if_label=True)
            
            # Cập nhật thêm task_dict theo ID thực tế để quản lý
            self.task_dict[next_id] = label_info['labels']
            
            return True
        return False

# Bổ sung vào class Client
    def get_feature_embeddings(self, model_to_use=None):
        """Trích xuất feature (output của layer trước FC) để vẽ t-SNE."""
        model = model_to_use if model_to_use is not None else self.model
        model.eval()
        
        # Lấy tất cả task đã học + task hiện tại
        history = self.tasks_seen_so_far + [self.current_task_id]
        test_datasets = [self.load_test_data(tid) for tid in history]
        loader = DataLoader(ConcatDataset(test_datasets), batch_size=self.batch_size, shuffle=False)
        
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                # Giả sử mô hình có phần 'base' trích xuất feature
                # Nếu dùng ResNet/CNN chuẩn, thường là model.base(x)
                f = model.base(x).view(x.size(0), -1) 
                feats.append(f.cpu())
                labels.append(y)
        return torch.cat(feats), torch.cat(labels)

    
    # def proto_eval(self, model, task, round):
    #     save_dir = os.path.join("pca_eval", self.file_name, "local")
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #
    #     # Save model state_dict
    #     model_filename = f"task_{task}_round_{round}.pth"
    #     model_path = os.path.join(save_dir, model_filename)
    #     torch.save(model.state_dict(), model_path)
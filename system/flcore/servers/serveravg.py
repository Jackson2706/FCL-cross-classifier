import time
import torch
import copy
import numpy as np
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k, read_client_data_FCL_cifar10

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # Gán lớp Client tương ứng
        self.set_clients(clientAVG)
        
        print(f"\nServer FedAvg Initialized. Join ratio: {self.join_ratio}")
        self.Budget = []

    def train(self):
        # Vòng lặp qua từng Task (Continual Learning)
        for task in range(self.args.num_tasks):
            print(f"\n================ Current Task: {task} =================")
            self.current_task = task
            
            # --- 1. SETUP DATA CHO TASK MỚI ---
            if task == 0:
                # Task đầu tiên: Chỉ cập nhật nhãn
                self._update_label_info()
            else:
                # Các Task sau: Load dữ liệu mới cho từng client
                self._load_task_data_for_clients(task)
                self._update_label_info()

            # --- 2. VÒNG LẶP COMMUNICATION ROUNDS ---
            for i in range(self.global_rounds):
                s_t = time.time()
                glob_iter = i + self.global_rounds * task
                
                # a. Chọn Clients và Gửi Model
                self.selected_clients = self.select_clients()
                self.send_models() 

                # b. Log Global Accuracy (Trước khi train)
                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                # c. Client Local Training
                for client in self.selected_clients:
                    client.train(task=task)

                # d. Log Local Accuracy (Sau khi train)
                if i % self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")

                # e. Aggregation (FedAvg)
                self.receive_models() 
                self.aggregate_parameters()
                
                self.Budget.append(time.time() - s_t)
                print(f"Round {i} Time: {self.Budget[-1]:.2f}s")

            # --- 3. ĐÁNH GIÁ CUỐI TASK (FORGETTING RATE) ---
            if self.args.offlog:
                print(f"\n>>> Task {task} Finished. Evaluating Forgetting Rate...")
                # Gửi model mới nhất xuống client để test
                self.send_models() 
                # Hàm eval_task trong ServerBase sẽ tính toán và log Forgetting
                self.eval_task(task=task, glob_iter=glob_iter, flag="global")

    def _update_label_info(self):
        """Cập nhật danh sách nhãn hiện có trong hệ thống"""
        available_labels = set()
        for u in self.clients:
            available_labels = available_labels.union(set(u.classes_so_far))
        self.available_labels_current = available_labels

    def _load_task_data_for_clients(self, task):
        """Helper để load dữ liệu task mới cho từng client"""
        # Chọn hàm đọc dữ liệu dựa trên config
        if 'IMAGENET1k' in self.args.dataset:
            read_func = read_client_data_FCL_imagenet1k
        elif 'CIFAR100' in self.args.dataset:
            read_func = read_client_data_FCL_cifar100
        else:
            read_func = read_client_data_FCL_cifar10

        for i, client in enumerate(self.clients):
            train_data, label_info = read_func(
                i, task=task, classes_per_task=self.args.cpt, count_labels=True
            )
            # Gọi hàm next_task của client để cập nhật dữ liệu
            client.next_task(train_data, label_info)
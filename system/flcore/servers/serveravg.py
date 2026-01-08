import copy
import statistics
import time

import numpy as np
import torch
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.lr_scheduler import StepLR
from utils.data_utils import *
from utils.model_utils import ParamDict


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for task in range(self.args.num_tasks):
            print(f"\n================ Current Task: {task} =================")
            for i in range(self.global_rounds):
                glob_iter = i + self.global_rounds * task
                s_t = time.time()
                self.selected_clients = self.select_clients()
                for client in self.selected_clients:
                    client.train(task=task)
                self.receive_models()
                self.receive_grads()
                self.aggregate_parameters()
                self.send_models()
                self.eval(task=task, glob_iter=i + task*self.global_rounds, flag="global")

            self.eval(task=task, glob_iter=i + task*self.global_rounds, flag="global")
            self.send_models()
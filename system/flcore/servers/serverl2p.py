import time
import torch
import copy
from flcore.clients.clientl2p import clientL2P
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
from utils.model_utils import ParamDict
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from torch.optim.lr_scheduler import StepLR
import numpy as np

import statistics


class FedL2P(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        for n, p in self.global_model.named_parameters():
            if ("prompt" in n and not "pos_embed" in n) or "last" in n:
                p.requires_grad = True
            else:
                p.requires_grad = True

        self.set_clients(clientL2P)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):

        if self.args.num_tasks % self.N_TASKS != 0:
            raise ValueError("Set num_task again")

        for task in range(self.args.num_tasks):

            print(f"\n================ Current Task: {task} =================")
            if task == 0:
                 # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            else:
                self.current_task = task
                
                torch.cuda.empty_cache()
                for i in range(len(self.clients)):
                    
                    if self.args.dataset == 'IMAGENET1k':
                        train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    elif self.args.dataset == 'CIFAR100':
                        train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    else:
                        raise NotImplementedError("Not supported dataset")

                    # update dataset
                    self.clients[i].next_task(train_data, label_info) # assign dataloader for new data
                    # print(self.clients[i].task_dict)

                # update labels info.
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                # print("ahihi" + str(available_labels_current))

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

                    # print(available_labels)

            # ============ train ==============

            for i in range(self.global_rounds):

                glob_iter = i + self.global_rounds * task
                s_t = time.time()
                self.selected_clients = self.select_clients()
                self.send_models()

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                for client in self.selected_clients:
                    client.train(task=task)

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]

                self.receive_models()
                self.receive_grads()
                model_origin = copy.deepcopy(self.global_model)
                self.aggregate_parameters()

                angle = [self.cos_sim(model_origin, self.global_model, models) for models in self.uploaded_models]
                distance = [self.distance(self.global_model, models) for models in self.uploaded_models]
                norm = [self.distance(model_origin, models) for models in self.uploaded_models]
                self.angle_value = statistics.mean(angle)
                self.distance_value = statistics.mean(distance)
                self.norm_value = statistics.mean(norm)
                angle_value = []
                for grad_i in self.grads:
                    for grad_j in self.grads:
                        angle_value.append(self.cosine_similarity(grad_i, grad_j))
                self.grads_angle_value = statistics.mean(angle_value)
                print(f"grad angle: {self.grads_angle_value}")


                if i%self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
                if self.args.offlog == True and not self.args.debug:  
                    self.eval_task(task=task, glob_iter=glob_iter, flag="local")

                    # need eval before data update
                    self.send_models()
                    self.eval_task(task=task, glob_iter=glob_iter, flag="global")

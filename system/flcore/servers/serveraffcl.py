import torch
import time
import numpy as np
import copy
import statistics

from flcore.clients.clientaffcl import ClientAFFCL
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k

class FedAFFCL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.classifier_head_list = ['classifier.fc_classifier', 'classifier.fc2']

        self.set_clients(ClientAFFCL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

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

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            
            # ============ train ==============

            for i in range(self.global_rounds):
                
                glob_iter = i + self.global_rounds * task
                s_t = time.time()
                self.selected_clients = self.select_clients()
                self.send_parameters(mode='all', beta=1)

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                global_classifier = self.global_model.classifier
                global_classifier.eval()

                for client in self.selected_clients:
                    verbose = False
                    client.train(task, glob_iter, global_classifier, verbose=verbose)

                self.receive_models()
                self.receive_grads()
                model_origin = copy.deepcopy(self.global_model)
                self.aggregate_parameters_affcl()

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

    def aggregate_parameters_affcl(self, class_partial=False):
        assert (self.selected_clients is not None and len(self.selected_clients) > 0)
        
        param_dict = {}
        for name, param in self.global_model.named_parameters():
            param_dict[name] = torch.zeros_like(param.data)
        
        total_train = 0
        for client in self.selected_clients:
            total_train += len(client.train_data) # length of the train data for weighted importance
        
        param_weight_sum = {}
        for client in self.selected_clients:
            for name, param in client.model.named_parameters():
                if ('fc_classifier' in name and class_partial):
                    class_available = torch.Tensor(client.classes_so_far).long()
                    param_dict[name][class_available] += param.data[class_available] * len(client.train_data) / total_train
                    
                    add_weight = torch.zeros([param.data.shape[0]]).cuda()
                    add_weight[class_available] = len(client.train_data) / total_train
                else:
                    param_dict[name] += param.data * len(client.train_data) / total_train
                    add_weight = len(client.train_data) / total_train
                
                if name not in param_weight_sum.keys():
                    param_weight_sum[name] = add_weight
                else:
                    param_weight_sum[name] += add_weight
                
        for name, param in self.global_model.named_parameters():

            if 'fc_classifier' in name and class_partial:
                valid_class = (param_weight_sum[name]>0)
                weight_sum = param_weight_sum[name][valid_class]
                if 'weight' in name:
                    weight_sum = weight_sum.view(-1, 1)
                param.data[valid_class] = param_dict[name][valid_class]/weight_sum
            else:
                param.data = param_dict[name]/param_weight_sum[name]

    def add_parameters(self, client, ratio, partial=False):
        if partial:
            for server_param, client_param in zip(self.global_model.get_shared_parameters(), client.model.get_shared_parameters()):
                server_param.data = server_param.data + client_param.data.clone() * ratio
        else:
            # replace all!
            for server_param, client_param in zip(self.global_model.parameters(), client.model.parameters()):
                server_param.data = server_param.data + client_param.data.clone() * ratio

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            
            if self.args.dataset == 'IMAGENET1k':
                train_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
            elif self.args.dataset == 'CIFAR100':
                train_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
            else:
                raise NotImplementedError("Not supported dataset")

            client = clientObj(self.args, id=i, train_data=train_data, classifier_head_list=self.classifier_head_list)
            self.clients.append(client)

            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])
            client.task_dict[0] = label_info['labels']

    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.clients
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users
        
        for user in users:
            if mode == 'all': # share all parameters
                user.set_parameters_precise(self.global_model, beta=beta)
            else: # share a part parameters
                user.set_shared_parameters(self.global_model, mode=mode)
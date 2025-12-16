import time
import torch
from flcore.clients.clientdbe import clientDBE
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
from threading import Thread

import statistics
import copy

class FedDBE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # initialization period
        self.set_clients(clientDBE)
        self.selected_clients = self.clients
        for client in self.selected_clients:
            client.train(task=0) # no DBE

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += len(client.train_data)
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(len(client.train_data))
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
            
        global_mean = 0
        for cid, w in zip(self.uploaded_ids, self.uploaded_weights):
            global_mean += self.clients[cid].running_mean * w
        print('>>>> global_mean <<<<', global_mean)
        for client in self.selected_clients:
            client.global_mean = global_mean.data.clone()

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        print('featrue map shape: ', self.clients[0].client_mean.shape)
        print('featrue map numel: ', self.clients[0].client_mean.numel())


    def train(self):

        # if self.args.num_tasks % self.N_TASKS != 0:
        #     raise ValueError("Set num_task again")

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

                if self.args.seval:
                    self.spatio_grad_eval(model_origin=model_origin, glob_iter=glob_iter)

                if self.args.pca_eval:
                    self.proto_eval(model=self.global_model, task=task, round=i)

                # if i%self.eval_gap == 0:
                #     self.eval(task=task, glob_iter=glob_iter, flag="local")

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            # if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
            #     if self.args.offlog == True and not self.args.debug:
            #         self.eval_task(task=task, glob_iter=glob_iter, flag="local")
                    
            #         # need eval before data update
            #         self.send_models()
            #         self.eval_task(task=task, glob_iter=glob_iter, flag="global")

import time
import copy
import numpy as np
import torch
# from flcore.clients.clientavg import clientAVG
from flcore.clients.clientas import clientAS
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k

import statistics

class FedAS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientAS)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def all_clients(self):
        return self.clients

    def send_selected_models(self, selected_ids, epoch, task):
        assert (len(self.clients) > 0)

        # for client in self.clients:
        for client in [client for client in self.clients if (client.id in selected_ids)]:
            start_time = time.time()

            progress = (epoch+1) / self.global_rounds
            
            client.set_parameters(self.global_model, progress, task)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)    
    
    def aggregate_wrt_fisher(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        # calculate the aggregrate weight with respect to the FIM value of model
        FIM_weight_list = []
        for id in self.uploaded_ids:
            FIM_weight_list.append(self.clients[id].fim_trace_history[-1])
        # normalization to obtain weight
        FIM_weight_list = [FIM_value/sum(FIM_weight_list) for FIM_value in FIM_weight_list]

        for w, client_model in zip(FIM_weight_list, self.uploaded_models):
            self.add_parameters(w, client_model)

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
                self.alled_clients = self.all_clients()

                selected_ids = [client.id for client in self.selected_clients]

                # self.send_models()
                self.send_selected_models(selected_ids, i, task)

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                # self.send_models()

                # print(f'send selected models done')

                # for client in self.selected_clients:
                #     client.train()

                for client in self.alled_clients:
                    # print("===============")
                    client.train(client.id in selected_ids, task)
                # assert 1==0

                self.print_fim_histories()

                self.receive_models()
                self.receive_grads()
                model_origin = copy.deepcopy(self.global_model)
                self.aggregate_wrt_fisher()

                if self.args.seval:
                    self.spatio_grad_eval(model_origin=model_origin)

                if self.args.pca_eval:
                    self.proto_eval(model=self.global_model, task=task, round=i)

                if i%self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
                if self.args.offlog == True and not self.args.debug: 
                    self.eval_task(task=task, glob_iter=glob_iter, flag="local")

                    # need eval before data update
                    self.send_selected_models(selected_ids, self.global_rounds-1, task)
                    self.eval_task(task=task, glob_iter=glob_iter, flag="global")

            # print(f'+++++++++++++++++++++++++++++++++++++++++')
            # gen_acc = self.avg_generalization_metrics()
            # print(f'Generalization Acc: {gen_acc}')
            # print(f'+++++++++++++++++++++++++++++++++++++++++')

    def print_fim_histories(self):
        avg_fim_histories = []

        # Print FIM trace history for each client
        # for client in self.selected_clients:
        for client in self.alled_clients:
            formatted_history = [f"{value:.1f}" for value in client.fim_trace_history]
            print(f"Client{client.id} : {formatted_history}")
            avg_fim_histories.append(client.fim_trace_history)

        # Calculate and print average FIM trace history across clients
        avg_fim_histories = np.mean(avg_fim_histories, axis=0)
        formatted_avg = [f"{value:.1f}" for value in avg_fim_histories]
        print(f"Avg Sum_T_FIM : {formatted_avg}")


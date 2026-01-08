import copy
import statistics
import time

import numpy as np
import torch
from flcore.clients.clientrefedplus import clientReFedPlus
from flcore.servers.serverbase import Server
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.lr_scheduler import StepLR
from utils.data_utils import *
from utils.model_utils import ParamDict


class ReFedPlus(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # Re-Fed+ specific parameters
        self.pim_iterations = args.pim_iterations if hasattr(args, 'pim_iterations') else 5  # s iterations
        self.lambda_param = args.lambda_param if hasattr(args, 'lambda_param') else 0.5  # λ for local-global tradeoff
        self.pim_lr = args.pim_lr if hasattr(args, 'pim_lr') else 0.01  # learning rate for PIM updates

        self.set_clients(clientReFedPlus)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(f"PIM iterations: {self.pim_iterations}, Lambda: {self.lambda_param}")
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
                print("ahihi " + str(len(available_labels_current)))
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            else:
                self.current_task = task

                torch.cuda.empty_cache()
                for i in range(len(self.clients)):

                    if self.args.dataset == 'IMAGENET1k':
                        train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task,
                                                                                 classes_per_task=self.args.cpt,
                                                                                 count_labels=True)
                    elif self.args.dataset == 'CIFAR100':
                        train_data, label_info = read_client_data_FCL_cifar100(i, task=task,
                                                                               classes_per_task=self.args.cpt,
                                                                               count_labels=True)
                    elif self.args.dataset == 'CIFAR10':
                        train_data, label_info = read_client_data_FCL_cifar10(i, task=task,
                                                                              classes_per_task=self.args.cpt,
                                                                              count_labels=True)
                    else:
                        raise NotImplementedError("Not supported dataset")

                    # update dataset
                    self.clients[i].next_task(train_data, label_info)  # assign dataloader for new data
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
            # # Train PIM before starting the task (Re-Fed+ specific)
            # if task > 0:
            #     print(f"\n--- Pre-training PIM for Task {task} ---")
            #     self._pretrain_pim(task)

            for i in range(self.global_rounds):

                glob_iter = i + self.global_rounds * task
                s_t = time.time()

                # Server randomly selects a subset of devices
                self.selected_clients = self.select_clients()
                self.send_models()

                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                for client in self.selected_clients:
                    client.train(task=task)

                # threads = [Thread(target=client.train)
                #            for client in self.selected_clients]
                # [t.start() for t in threads]
                # [t.join() for t in threads]

                self.receive_models()

                # Cache important samples for incremental learning
                for client in self.selected_clients:
                    client.cache_important_samples(self.global_model)

                self.aggregate_parameters()

                if i % self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")

                self.Budget.append(time.time() - s_t)
                print(f"\n{'=' * 50}")
                print(f"Re-Fed+ Training Complete!")
                print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            # End of task evaluation
            if int(task / self.N_TASKS) == int(self.args.num_tasks / self.N_TASKS - 1):
                if self.args.offlog == True and not self.args.debug:
                    self.eval_task(task=task, glob_iter=glob_iter, flag="local")

                    # need eval before data update
                    self.send_models()
                    self.eval_task(task=task, glob_iter=glob_iter, flag="global")

    def _pretrain_pim(self, task):
        """
        Pre-train PIM (Personalized Informative Model) before new task arrival
        This is a lightweight process consuming 1/40 of standard task training resources
        """
        print("Pre-training PIM on historical data...")

        for client in self.selected_clients:
            # Update PIM using historical data before new task processing
            # This happens after client receives global model w^{t-1}
            client.update_pim_before_task(
                global_model=self.global_model,
                pim_iterations=self.pim_iterations,
                lambda_param=self.lambda_param,
                learning_rate=self.pim_lr
            )

        print("PIM pre-training completed.")
import time
import torch
import copy
from flcore.clients.clientstgm import clientSTGM
from flcore.servers.serverbase import Server
from utils.data_utils import *
from utils.model_utils import ParamDict
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from torch.optim.lr_scheduler import StepLR
import numpy as np

import statistics


class FedSTGM(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientSTGM)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.stgm_learning_rate = args.stgm_learning_rate
        self.stgm_step_size = args.stgm_step_size
        self.stgm_c = args.stgm_c
        self.stgm_rounds = args.stgm_rounds
        self.stgm_momentum = args.stgm_momentum
        self.stgm_gamma = args.stgm_gamma

        self.stgm_meta_lr = args.stgm_meta_lr
        self.grad_balance = args.grad_balance

    def train(self):

        if self.args.num_tasks % self.N_TASKS != 0:
            raise ValueError("Set num_task again")

        for task in range(self.args.num_tasks):
        # for task in range(self.N_TASKS):

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
                        train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt,
                                                                                 count_labels=True)
                    elif self.args.dataset == 'CIFAR100':
                        train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt,
                                                                               count_labels=True)
                    elif self.args.dataset == 'CIFAR10':
                        train_data, label_info = read_client_data_FCL_cifar10(i, task=task, classes_per_task=self.args.cpt,
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

                    # print(available_labels)

            # ============ train ==============

            for i in range(self.global_rounds):

                glob_iter = i + self.global_rounds * task
                s_t = time.time()
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

                """
                Spatio Gradient Matching
                """
                self.receive_models()
                self.receive_grads()
                if self.args.sgm:
                    """
                    Version 1

                    """
                    # self.receive_grads()
                    # grad_ez = sum(p.numel() for p in self.global_model.parameters())
                    # grads = torch.Tensor(grad_ez, self.num_clients)
                    # print(f"size of grads: {grads.size()}")
                    #
                    # for index, model in enumerate(self.grads):
                    #     grad2vec2(model, grads, index)
                    #
                    # g = self.aggregate_stgm(grads, self.num_clients)
                    #
                    #
                    # # model_origin = copy.deepcopy(self.global_model)
                    # self.overwrite_grad2(self.global_model, g)
                    # for param in self.global_model.parameters():
                    #     param.data += param.grad


                    """
                    Version 2
                    
                    - flatten_meta_weights += g * lr_meta
                    - vector_to_parameters(flatten_meta_weights, meta_weights.parameters())
                    - meta_weights = ParamDict(meta_weights.state_dict())
                    """
                    model_origin = copy.deepcopy(self.global_model)
                    meta_weights = self.stgm_high(
                        meta_weights=self.global_model,
                        inner_weights=self.uploaded_models,
                        lr_meta= self.stgm_meta_lr
                    )
                    self.global_model.load_state_dict(copy.deepcopy(meta_weights))

                else:
                    model_origin = copy.deepcopy(self.global_model)
                    self.aggregate_parameters()

                if self.args.seval:
                    self.spatio_grad_eval(model_origin = model_origin)

                if task > 10:
                    if self.args.pca_eval:
                        self.proto_eval(global_model=self.global_model,
                                        local_model=self.uploaded_models[0], task=task, round=i)

                self.Budget.append(time.time() - s_t)
                print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
                # if i % self.eval_gap == 0:
                #     self.eval(task=task, glob_iter=glob_iter, flag="local")

            if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
                if not self.args.debug:
                    self.eval_task(task=task, glob_iter=glob_iter, flag="local")

                    # need eval before data update
                    self.send_models()
                    self.eval_task(task=task, glob_iter=glob_iter, flag="global")

    def stgm_high(self, meta_weights, inner_weights, lr_meta):
        """
        Input:
        - meta_weights: class X(nn.Module)
        - inner_weights: list[X(nn.Module), X(nn.Module), ..., X(nn.Module)]
        - lr_meta: scalar value

        Output:
        - meta_weights: class X(nn.Module)

        """
        all_domain_grads = []
        flatten_meta_weights = torch.cat([param.view(-1) for param in meta_weights.parameters()])
        for i_domain in range(self.num_clients):
            domain_grad_diffs = [torch.flatten(inner_param - meta_param) for inner_param, meta_param in
                                 zip(inner_weights[i_domain].parameters(), meta_weights.parameters())]
            domain_grad_vector = torch.cat(domain_grad_diffs)
            all_domain_grads.append(domain_grad_vector)

        """
        - Grads normalization.
        """
        if self.grad_balance:
            # Apply balancing
            # Step 1: Compute norms for each gradient vector
            domain_grad_norms = [torch.norm(grad) for grad in all_domain_grads]

            # Step 2: Determine scaling factors to balance the norms
            # Example: Scale all norms to a target value (e.g., the average norm)
            target_norm = torch.mean(torch.tensor(domain_grad_norms))
            scaling_factors = [target_norm / norm if norm > 0 else 1.0 for norm in domain_grad_norms]

            # Step 3: Scale gradient vectors
            balanced_retain_grads = [grad * scale for grad, scale in zip(domain_grad_norms, scaling_factors)]

            # Step 4: Stack the balanced gradients into a tensor
            all_domains_grad_tensor = torch.stack(balanced_retain_grads).t()
        else:
            all_domains_grad_tensor = torch.stack(all_domain_grads).t()

        all_domains_grad_tensor = torch.stack(all_domain_grads).t()

        # print(all_domains_grad_tensor)
        g = self.stgm_low(all_domains_grad_tensor, self.num_clients)

        flatten_meta_weights += g * lr_meta

        vector_to_parameters(flatten_meta_weights, meta_weights.parameters())
        meta_weights = ParamDict(meta_weights.state_dict())

        return meta_weights

    def stgm_low(self, grad_vec, num_tasks):

        grads = grad_vec.to(self.device)

        GG = grads.t().mm(grads)
        # to(device)
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True, device=self.device)
        #         w = torch.zeros(num_tasks, 1, requires_grad=True).to(self.device)

        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=self.stgm_learning_rate * 2, momentum=self.stgm_momentum)
        else:
            w_opt = torch.optim.SGD([w], lr=self.stgm_learning_rate, momentum=self.stgm_momentum)

        scheduler = StepLR(w_opt, step_size=self.stgm_step_size, gamma=self.stgm_gamma)

        c = (gg + 1e-4).sqrt() * self.stgm_c

        w_best = None
        obj_best = np.inf
        for i in range(self.stgm_rounds + 1):
            w_opt.zero_grad()
            ww = torch.softmax(w, dim=0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < self.stgm_rounds:
                obj.backward(retain_graph=True)
                w_opt.step()
                scheduler.step()
        
        ww = torch.softmax(w_best, dim=0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm + 1e-4)
        g = ((1 / num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads.t()).sum(0) / (1 + self.stgm_c ** 2)
        return g

    def aggregate_stgm(self, grad_vec, num_tasks):
        grads = grad_vec.to(self.device)

        GG = grads.t().mm(grads)
        print(f"size GG: {GG.size()}")
        # to(device)
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True, device=self.device)
        #         w = torch.zeros(num_tasks, 1, requires_grad=True).to(self.device)

        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=self.stgm_learning_rate * 2, momentum=self.stgm_momentum)
        else:
            w_opt = torch.optim.SGD([w], lr=self.stgm_learning_rate, momentum=self.stgm_momentum)

        scheduler = StepLR(w_opt, step_size=self.stgm_step_size, gamma=self.stgm_momentum)

        c = (gg + 1e-4).sqrt() * self.stgm_c

        w_best = None
        obj_best = np.inf
        for i in range(self.stgm_rounds + 1):
            w_opt.zero_grad()
            ww = torch.softmax(w, dim=0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < self.stgm_rounds:
                obj.backward()
                w_opt.step()
                scheduler.step()

                # Check this scheduler. step()

        ww = torch.softmax(w_best, dim=0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm + 1e-4)
        g = ((1 / num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads.t()).sum(0) / (1 + self.stgm_c ** 2)
        return g

    def overwrite_grad2(self, m, newgrad):
        newgrad = newgrad * self.num_clients
        for param in m.parameters():
            # Get the number of elements in the current parameter
            num_elements = param.numel()

            # Extract a slice of new_params with the same number of elements
            param_slice = newgrad[:num_elements]

            # Reshape the slice to match the shape of the current parameter
            param.grad = param_slice.view(param.data.size())

            # Move to the next slice in new_params
            newgrad = newgrad[num_elements:]

def grad2vec2(m, grads, task):
    grads[:, task].fill_(0.0)
    all_params = torch.cat([param.detach().view(-1) for param in m.parameters()])
    # print(all_params.size())
    grads[:, task].copy_(all_params)
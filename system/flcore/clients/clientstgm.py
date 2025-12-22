import numpy as np
import time
import copy
import statistics
import torch
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import StepLR
from utils.model_utils import ParamDict
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from flcore.clients.clientbase import Client

from flcore.trainmodel.models import *

from flcore.utils_core.buffer_utils import ImagePool
from flcore.utils_core.protonet_utils import ProtoNet_Loss, OnPro_Loss


class clientSTGM(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        self.tgm_learning_rate = args.tgm_learning_rate
        self.tgm_step_size = args.tgm_step_size
        self.tgm_c = args.tgm_c
        self.tgm_rounds = args.tgm_rounds
        self.tgm_momentum = args.tgm_momentum
        self.tgm_gamma = args.tgm_gamma

        self.tgm_meta_lr = args.tgm_meta_lr

        self.grad_balance = args.grad_balance

        # self.mem_manager = ImagePool(root=self.save_dir)

        self.proto_loss = ProtoNet_Loss(n_support = 20)

        """ ==== Optimizer for ProtoNet ==== """
        if args.optimizer == "sgd":
            self.optimizer_proto = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
            self.optimizer_head = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        elif args.optimizer == "adam":
            self.optimizer_proto = torch.optim.Adam(self.model.base.parameters(), lr=self.learning_rate)
            self.optimizer_head = torch.optim.SGD(self.model.head.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}.")
        self.lr_scheduler_proto = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_proto,
            gamma=args.learning_rate_decay_gamma
        )
        self.lr_scheduler_head = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_head,
            gamma=args.learning_rate_decay_gamma
        )

    def train(self, task=None):
        trainloader = self.load_train_data(task=task)
        self.model.train()
        old_model = copy.deepcopy(self.model)

        start_time = time.time()

        self.optimizer_inner_state = self.optimizer.state_dict()
        self.optimizer_proto_inner_state = self.optimizer_proto.state_dict()
        self.optimizer_head_inner_state = self.optimizer_head.state_dict()

        max_local_epochs = self.local_epochs
        """ ============ Current Task ============  """
        if self.args.coreset:   # TODO: Use coreset -> ProtoNet + Classifier
            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    # TODO First Step: ProtoNet update
                    proto, output = self.model.get_proto(x)
                    proto_metric = self.proto_loss(proto, y)
                    self.optimizer_proto.zero_grad()
                    proto_metric[0].backward()
                    self.optimizer_proto.step()

                    # TODO Second Step: Entire model update (Or classifier only?)
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer_head.zero_grad()
                    loss.backward()
                    self.optimizer_head.step()
        else:       # TODO: Do not use coreset -> normal learner
            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    # TODO Base+Head use the same classification loss
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


        if self.args.tgm and (self.current_task >= 0):
            # inner_models = [copy.deepcopy(self.model)]

            """ ======== Approximate Last Task ========  """
            self.create_clone(n_task=len(self.task_dict) - 1)

            """ ==== Train CoreSet ==== """
            # FIXME Split support/query set
            for task_id, task in enumerate(self.task_dict):
                if task_id == self.current_task:
                    pass
                else:
                    if self.args.coreset:
                        # TODO Load CoreSet
                        trainloader = self.load_train_data(task=task)
                        for epoch in range(max_local_epochs):
                            for i, (x, y) in enumerate(trainloader):
                                if type(x) == type([]):
                                    x[0] = x[0].to(self.device)
                                else:
                                    x = x.to(self.device)
                                y = y.to(self.device)

                                # TODO First Step: ProtoNet update
                                proto, output = self.model.get_proto(x)
                                proto_metric = self.proto_loss(proto, y)
                                self.optimizer_proto_inner[task_id].zero_grad()
                                proto_metric[0].backward()
                                self.optimizer_proto_inner[task_id].step()

                                # TODO Second Step: Entire model update (Or classifier only?)
                                output = self.model(x)
                                loss = self.loss(output, y)
                                self.optimizer_head_inner[task_id].zero_grad()
                                loss.backward()
                                self.optimizer_head_inner[task_id].step()

                    else: # TODO Base+Head use the same classification loss
                        trainloader = self.load_train_data(task=task)
                        for epoch in range(max_local_epochs):
                            for i, (x, y) in enumerate(trainloader):
                                if type(x) == type([]):
                                    x[0] = x[0].to(self.device)
                                else:
                                    x = x.to(self.device)
                                y = y.to(self.device)
                                # TODO Base+Head use the same classification loss
                                output = self.network_inner[task_id](x)
                                loss = self.loss(output, y)
                                self.optimizer_inner[task_id].zero_grad()
                                loss.backward()
                                self.optimizer_inner[task_id].step()

            self.network_inner.append(self.model)

            """ ===== Temporal Gradient Matching ======  """
            meta_weights = self.tgm_high(
                meta_weights=old_model,
                inner_weights=self.network_inner,
                lr_meta=2*self.tgm_meta_lr
            )
            self.model.load_state_dict(copy.deepcopy(meta_weights))
        else:
            pass

        # if self.args.teval:
        #     self.grad_eval(old_model=self.model)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def tgm_high(self, meta_weights, inner_weights, lr_meta):
        """
        Input:
        - meta_weights: class X(nn.Module)
        - inner_weights: list[X(nn.Module), X(nn.Module), ..., X(nn.Module)]
        - lr_meta: scalar value

        Output:
        - meta_weights: class X(nn.Module)

        """
        all_domain_grads = []
        num_tasks = len(inner_weights)
        flatten_meta_weights = torch.cat([param.view(-1) for param in meta_weights.parameters()])
        for i_domain in range(num_tasks):
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
        g = self.tgm_low(all_domains_grad_tensor, num_tasks)

        flatten_meta_weights += g * lr_meta

        vector_to_parameters(flatten_meta_weights, meta_weights.parameters())
        meta_weights = ParamDict(meta_weights.state_dict())

        return meta_weights

    def tgm_low(self, grad_vec, num_tasks):

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
            w_opt = torch.optim.SGD([w], lr=self.tgm_learning_rate * 2, momentum=self.tgm_momentum)
        else:
            w_opt = torch.optim.SGD([w], lr=self.tgm_learning_rate, momentum=self.tgm_momentum)

        scheduler = StepLR(w_opt, step_size=self.tgm_step_size, gamma=self.tgm_gamma)

        c = (gg + 1e-4).sqrt() * self.tgm_c

        w_best = None
        obj_best = np.inf
        for i in range(self.tgm_rounds + 1):
            w_opt.zero_grad()
            ww = torch.softmax(w, dim=0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < self.tgm_rounds:
                obj.backward(retain_graph=True)
                w_opt.step()
                scheduler.step()

                # Check this scheduler. step()

        ww = torch.softmax(w_best, dim=0)
        gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm + 1e-4)
        g = ((1 / num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads.t()).sum(0) / (1 + self.tgm_c ** 2)
        return g

    def create_clone(self, n_task):
        self.network_inner = []
        self.optimizer_inner = []
        self.optimizer_proto_inner = []
        self.optimizer_head_inner = []
        
        for task_id in range(n_task):
            temp_model = FedAvgCNN(in_features=3, num_classes=self.num_classes, dim=1600).to(self.device)
            temp_head = copy.deepcopy(temp_model.fc)
            temp_model.fc = nn.Identity()
            temp_model = BaseHeadSplit(temp_model, temp_head)

            temp_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            self.network_inner.append(temp_model)

            if self.args.optimizer == "sgd":
                self.optimizer_inner.append(
                    torch.optim.SGD(self.network_inner[task_id].parameters(), lr=self.learning_rate)
                )
                self.optimizer_proto_inner.append(
                    torch.optim.SGD(self.network_inner[task_id].base.parameters(), lr=self.learning_rate)
                )
                self.optimizer_head_inner.append(
                    torch.optim.SGD(self.network_inner[task_id].head.parameters(), lr=self.learning_rate)
                )
            elif self.args.optimizer == "adam":
                self.optimizer_inner.append(
                    torch.optim.Adam(self.network_inner[task_id].parameters(), lr=self.learning_rate)
                )
                self.optimizer_proto_inner.append(
                    torch.optim.Adam(self.network_inner[task_id].base.parameters(), lr=self.learning_rate)
                )
                self.optimizer_head_inner.append(
                    torch.optim.Adam(self.network_inner[task_id].head.parameters(), lr=self.learning_rate)
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.args.optimizer}.")

            if self.optimizer_inner_state is not None:
                self.optimizer_inner[task_id].load_state_dict(self.optimizer_inner_state)
                self.optimizer_proto_inner[task_id].load_state_dict(self.optimizer_proto_inner_state)
                self.optimizer_head_inner[task_id].load_state_dict(self.optimizer_head_inner_state)

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
import copy
import torch
import torch.nn as nn
import numpy as np
import statistics
from torch.utils.data import DataLoader
from utils.data_utils import *

from flcore.trainmodel.models import *

import os
import joblib


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_data, **kwargs):
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
        self.last_copy.to(self.device)
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

    def _initialize_first_task(self):
        """Initialize labels information for the first task"""
        available_labels = set()
        available_labels_current = set()
        available_labels_past = set()

        for client in self.clients:
            available_labels.update(client.classes_so_far)
            available_labels_current.update(client.current_labels)

        for client in self.clients:
            client.available_labels = sorted(list(available_labels))
            client.available_labels_current = sorted(list(available_labels_current))
            client.available_labels_past = sorted(list(available_labels_past))
            # Initialize PIM for each client
            client.initialize_pim(self.global_model)

        print(f"Task 0 - Total labels: {len(available_labels)}, "
              f"Current: {len(available_labels_current)}, Past: {len(available_labels_past)}")

    def _prepare_new_task(self, task):
        """Prepare clients for a new task by loading new data"""
        self.current_task = task
        torch.cuda.empty_cache()

        for i in range(len(self.clients)):
            # Load new task data based on dataset
            if self.args.dataset == 'IMAGENET1k':
                train_data, label_info = read_client_data_FCL_imagenet1k(
                    i, task=task, classes_per_task=self.args.cpt, count_labels=True)
            elif self.args.dataset == 'CIFAR100':
                train_data, label_info = read_client_data_FCL_cifar100(
                    i, task=task, classes_per_task=self.args.cpt, count_labels=True)
            elif self.args.dataset == 'CIFAR10':
                train_data, label_info = read_client_data_FCL_cifar10(
                    i, task=task, classes_per_task=self.args.cpt, count_labels=True)
            else:
                raise NotImplementedError(f"Dataset {self.args.dataset} not supported")

            # Update client dataset
            self.clients[i].next_task(train_data, label_info)

        # Update labels information across all clients
        self._update_labels_info(task)

    def _update_labels_info(self, task):
        """Update available labels across all clients"""
        available_labels = set()
        available_labels_current = set()

        # Get past labels from first client (before update)
        available_labels_past = set(self.clients[0].available_labels) if task > 0 else set()

        # Collect all labels
        for client in self.clients:
            available_labels.update(client.classes_so_far)
            available_labels_current.update(client.current_labels)

        # Update all clients with new label information
        for client in self.clients:
            client.available_labels = sorted(list(available_labels))
            client.available_labels_current = sorted(list(available_labels_current))
            client.available_labels_past = sorted(list(available_labels_past))

        print(f"Task {task} - Total labels: {len(available_labels)}, "
              f"Current: {len(available_labels_current)}, Past: {len(available_labels_past)}")

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

        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)  

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def set_parameters_(self, model, only_critic, beta=1, mode=None, gr=False, classifier=None):
        '''
        At the beginning of this round:
        self.model: old user model, note trained yet
        model: the global model on the server (new model)
        '''
        if gr == True:
            for old_param, new_param in zip(self.classifier.critic.parameters(), classifier.critic.parameters()):
                if beta == 1:
                    old_param.data = new_param.data.clone()
                else:
                    old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()

        else:
            if only_critic == True:
                for old_param, new_param in zip(self.local_generator.critic.parameters(), model.critic.parameters()):
                    if beta == 1:
                        old_param.data = new_param.data.clone()
                    else:
                        old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()

            else:
                for old_param, new_param in zip(self.local_generator.parameters(), model.parameters()):
                    if beta == 1:
                        old_param.data = new_param.data.clone()
                    else:
                        old_param.data = beta * new_param.data.clone() + (1 - beta) * old_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self, task):
        testloader = self.load_test_data(task=task)

        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        
        return test_acc, test_num

    def test_metrics_(self, task):

        if self.args.model == "FedCIL":
            model = self.local_generator.critic
        else:
            model = self.model

        testloader = self.load_test_data(task=task)

        model.eval()

        test_acc = 0
        test_num = 0

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        return test_acc, test_num

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
        for model_i in network_test:
            for model_j in network_test:
                angle_value.append(self.cos_sim(old_model, model_i, model_j))
        self.t_angle_after = statistics.mean(angle_value)
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

    # def proto_eval(self, model, task, round):
    #     save_dir = os.path.join("pca_eval", self.file_name, "local")
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #
    #     # Save model state_dict
    #     model_filename = f"task_{task}_round_{round}.pth"
    #     model_path = os.path.join(save_dir, model_filename)
    #     torch.save(model.state_dict(), model_path)
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from torch.autograd import Variable


class clientDBE(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        self.klw = args.kl_weight
        self.momentum = args.momentum
        self.global_mean = None

        trainloader = self.load_train_data(task=self.current_task)        
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.running_mean = torch.zeros_like(rep[0])
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)

        self.client_mean = nn.Parameter(Variable(torch.zeros_like(rep[0])))
        self.opt_client_mean = torch.optim.SGD([self.client_mean], lr=self.learning_rate)


    def train(self, task):
        trainloader = self.load_train_data(task=task)
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs

        self.reset_running_stats()
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # ====== begin
                rep = self.model.base(x)
                running_mean = torch.mean(rep, dim=0)

                if self.num_batches_tracked is not None:
                    self.num_batches_tracked.add_(1)

                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * running_mean
                
                if self.global_mean is not None:
                    reg_loss = torch.mean(0.5 * (self.running_mean - self.global_mean)**2)
                    output = self.model.head(rep + self.client_mean)
                    loss = self.loss(output, y)
                    loss = loss + reg_loss * self.klw
                else:
                    output = self.model.head(rep)
                    loss = self.loss(output, y)
                # ====== end

                self.opt_client_mean.zero_grad()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.opt_client_mean.step()
                self.detach_running()

        # self.model.cpu()
        self.grad_eval(old_model=self.model)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def reset_running_stats(self):
        self.running_mean.zero_()
        self.num_batches_tracked.zero_()

    def detach_running(self):
        self.running_mean.detach_()

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
                rep = self.model.base(x)
                output = self.model.head(rep + self.client_mean)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def test_metrics(self, task):
        testloaderfull = self.load_test_data(task=task)
        self.model.eval()

        test_acc = 0
        test_num = 0
        reps = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep + self.client_mean)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                reps.extend(rep.detach())

        return test_acc, test_num

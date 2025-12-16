import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.utils_core.ALA import ALA


class clientALA(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx

        train_data = self.train_data
        self.ALA = ALA(self.id, self.loss, train_data, self.batch_size, 
                    self.rand_percent, self.layer_idx, self.eta, self.device)

    def train(self, task):
        trainloader = self.load_train_data(task=task)
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.args.teval:
            self.grad_eval(old_model=self.model)

        if self.args.pca_eval:
            self.proto_eval(model = self.model)

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        

    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)
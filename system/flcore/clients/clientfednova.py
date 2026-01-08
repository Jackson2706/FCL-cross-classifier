import time

from flcore.clients.clientbase import Client


class clientNova(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        self.train_steps = 0  # To store tau_i

    def train(self, task=None):
        trainloader = self.load_train_data(task=task)
        self.model.train()
        
        start_time = time.time()
        
        # Reset step counter for this round
        self.train_steps = 0

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

                # Count the step
                self.train_steps += 1
                
            self.learning_rate_scheduler.step()

        if self.args.teval:
            self.grad_eval(old_model=self.model)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
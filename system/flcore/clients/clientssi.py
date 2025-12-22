import torch
import torch.nn as nn
import time
import copy
from flcore.clients.clientbase import Client


class clientSSI(Client):
    """
    FedSSI client implementing surrogate model update (Eq. 5),
    parameter contribution estimation (Eq. 6),
    and target model training with surrogate regularization (Eq. 2).
    """

    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        self.lambda_q = args.lambda_q  # λ for q(λ)
        self.alpha = args.alpha        # α for L_total regularization
        # self.num_tasks = args.num_tasks
        self.v_model = self._init_surrogate_model()  # surrogate model v_k
        self.param_contrib = {}  # s_k^t
        self.param_importance = {}  # Ω_k^t

    def _init_surrogate_model(self):
        """Initialize the surrogate model v_k as a copy of the base model."""
        v_model = copy.deepcopy(self.model)
        self.clone_model(self.model, v_model)
        return v_model

    def surrogate_update(self, prev_global_model, dataloader, eta, s_iter=1):
        """
        Update surrogate model v_k^{t-1} using Eq. (5).
        """
        q_lambda = (1 - self.lambda_q) / (2 * self.lambda_q)

        self.v_model.train()
        criterion = nn.CrossEntropyLoss()
        v_opt = torch.optim.SGD(self.v_model.parameters(), lr=eta)

        for s in range(s_iter):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                v_opt.zero_grad()
                output = self.v_model(x)
                loss = criterion(output, y)
                loss.backward()

                # Gradient descent step with FedSSI adjustment term
                with torch.no_grad():
                    for v_param, w_param in zip(self.v_model.parameters(),
                                                prev_global_model.parameters()):
                        v_param -= eta * v_param.grad \
                                   + q_lambda * (v_param - w_param)

        return self.v_model

    def compute_param_contribution(self, loss_fn, dataloader):
        """
        Compute s_k^l (Eq. 6): parameter contribution over local updates.
        """
        self.param_contrib = {}
        self.v_model.train()

        for name, param in self.v_model.named_parameters():
            self.param_contrib[name] = torch.zeros_like(param)

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.v_model.zero_grad()
            loss = loss_fn(self.v_model(x), y)
            loss.backward()

            for name, param in self.v_model.named_parameters():
                if param.grad is not None:
                    self.param_contrib[name] += param.grad.detach().clone()

        # Normalize contributions
        for name in self.param_contrib:
            self.param_contrib[name] /= len(dataloader)

        return self.param_contrib

    def train(self, task=None, prev_global_model=None):
        """
        Full local training for current task t as in Algorithm 1.
        """
        trainloader = self.load_train_data(task=task)
        self.model.train()
        start_time = time.time()

        # ===== Step 1: Surrogate update (Eq. 5) =====
        self.surrogate_update(prev_global_model, trainloader, self.learning_rate, s_iter=self.local_epochs)

        # ===== Step 2: Compute parameter contribution (Eq. 6) =====
        loss_fn = nn.CrossEntropyLoss()
        self.compute_param_contribution(loss_fn, trainloader)

        # ===== Step 3: Train target model with surrogate regularization (Eq. 2) =====
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                L_new = loss_fn(output, y)

                # Surrogate regularization
                L_sur = 0.0
                for (name, param), (_, prev_param) in zip(self.model.named_parameters(),
                                                          prev_global_model.named_parameters()):
                    omega = self.param_importance.get(name, torch.ones_like(param))
                    L_sur += torch.sum(omega * (param - prev_param) ** 2)

                loss = L_new + self.alpha * L_sur
                loss.backward()
                optimizer.step()

        # ===== Step 4: bookkeeping =====
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return self.model

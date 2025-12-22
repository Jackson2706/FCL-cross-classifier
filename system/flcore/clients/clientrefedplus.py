# ==============================================================================
# File: clientrefedplus.py
# Description: Client-side implementation for Re-Fed+ algorithm
# ==============================================================================

import numpy as np
import time
import torch
import torch.nn as nn
import copy
from flcore.clients.clientbase import Client
from torch.utils.data import DataLoader, TensorDataset


class clientReFedPlus(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        # PIM (Personalized Informative Model) for this client
        self.pim_model = None
        self.pim_optimizer = None

        # Sample importance storage
        self.sample_importance_scores = {}  # Store importance scores for historical samples
        self.cached_samples = []  # Store cached samples with high importance
        self.max_cache_size = args.cache_size if hasattr(args, 'cache_size') else 200

        # Gradient norms tracking during PIM updates
        self.gradient_norms_history = []

        # Lambda parameter for local-global tradeoff
        self.lambda_param = args.lambda_param if hasattr(args, 'lambda_param') else 0.5

    def initialize_pim(self, global_model):
        """Initialize PIM model as a copy of the global model"""
        self.pim_model = copy.deepcopy(global_model)
        self.pim_model.to(self.device)

    def update_pim_before_task(self, global_model, pim_iterations, lambda_param, learning_rate):
        """
        Update PIM v_k^{t-1} through s iterations on historical data
        This implements equations (3), (4), and (5) from the paper

        Args:
            global_model: The global model w^{t-1}
            pim_iterations: Number of iterations s for PIM update
            lambda_param: λ parameter for local-global information tradeoff
            learning_rate: Learning rate η for PIM updates
        """
        if self.pim_model is None:
            self.initialize_pim(global_model)

        # Get historical data (data from previous tasks)
        historical_loader = self.load_historical_data()

        if historical_loader is None or len(historical_loader.dataset) == 0:
            print(f"Client {self.id}: No historical data available")
            return

        self.pim_model.train()

        # Setup optimizer for PIM
        pim_optimizer = torch.optim.SGD(self.pim_model.parameters(), lr=learning_rate)

        # Store global model parameters for regularization term
        global_params = {name: param.clone().detach()
                         for name, param in global_model.named_parameters()}

        # Calculate q(λ) = (1-λ)/(2λ) for balancing coefficient
        q_lambda = (1 - lambda_param) / (2 * lambda_param)

        print(f"Client {self.id}: Updating PIM for {pim_iterations} iterations")

        # Track gradient norms for importance scoring
        sample_gradient_norms = {}

        # Update PIM through s iterations
        for iteration in range(pim_iterations):
            epoch_gradient_norms = []

            for batch_idx, (x, y) in enumerate(historical_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                pim_optimizer.zero_grad()

                # Forward pass
                output = self.pim_model(x)

                # Calculate loss on historical data
                loss_data = self.loss(output, y)

                # Calculate regularization term: g(λ)(v_{k,s-1}^{t-1} - w^{t-1})
                regularization = 0.0
                for name, param in self.pim_model.named_parameters():
                    if name in global_params:
                        regularization += torch.sum((param - global_params[name]) ** 2)

                # Total loss: Equation (3)
                # v_{k,s}^{t-1} = v_{k,s-1}^{t-1} - η(∑∇l(...) + g(λ)(v_{k,s-1}^{t-1} - w^{t-1}))
                total_loss = loss_data + q_lambda * regularization

                # Backward pass
                total_loss.backward()

                # Calculate gradient norms for importance scoring (Equation 4)
                # G^p(x̃_{k,t-1}^{(i)}) = ||∇l(f_{v_{k,p}^{t-1}}(x̃_{k,t-1}^{(i)}), ỹ_{k,t-1}^{(i)})||^2
                for i in range(x.size(0)):
                    sample_idx = batch_idx * x.size(0) + i

                    # Calculate per-sample gradient norm
                    grad_norm = 0.0
                    for param in self.pim_model.parameters():
                        if param.grad is not None:
                            grad_norm += torch.sum(param.grad ** 2).item()

                    grad_norm = np.sqrt(grad_norm / x.size(0))  # Normalize by batch size
                    epoch_gradient_norms.append(grad_norm)

                    # Store gradient norm for this sample at this iteration
                    if sample_idx not in sample_gradient_norms:
                        sample_gradient_norms[sample_idx] = []
                    sample_gradient_norms[sample_idx].append(grad_norm)

                pim_optimizer.step()

        # Calculate importance scores using time-discounted accumulation (Equation 5)
        # I(x̃_{k,t-1}^{(i)}) = ∑_{p=1}^{s} (1/p)G^p(x̃_{k,t-1}^{(i)})
        for sample_idx, grad_norms in sample_gradient_norms.items():
            importance_score = 0.0
            for p, grad_norm in enumerate(grad_norms, start=1):
                importance_score += (1.0 / p) * grad_norm

            self.sample_importance_scores[sample_idx] = importance_score

        print(f"Client {self.id}: PIM update completed. "
              f"Computed importance scores for {len(self.sample_importance_scores)} samples")

    def cache_important_samples(self, historical_loader):
        """
        Cache previous samples with higher importance scores
        This implements line 9 of Algorithm 1
        """
        if len(self.sample_importance_scores) == 0:
            return

        # Sort samples by importance score (descending)
        sorted_samples = sorted(self.sample_importance_scores.items(),
                                key=lambda x: x[1], reverse=True)

        # Cache top samples up to max_cache_size
        self.cached_samples = []

        # Get the actual samples from historical data
        historical_data = historical_loader.dataset

        for sample_idx, importance_score in sorted_samples[:self.max_cache_size]:
            if sample_idx < len(historical_data):
                x, y = historical_data[sample_idx]
                self.cached_samples.append((x, y, importance_score))

        print(f"Client {self.id}: Cached {len(self.cached_samples)} important samples")

    def train(self, task=None, round_num=None):
        """
        Training the local model with cached samples and new task data
        This implements lines 10-11 of Algorithm 1

        Args:
            task: Current task number
            round_num: Current communication round
        """
        # Load new task data
        trainloader = self.load_train_data(task=task)
        self.model.train()

        start_time = time.time()
        max_local_epochs = self.local_epochs

        # Before training, update PIM and cache important samples if we have historical data
        if task > 0 and round_num == 0:
            # This happens at the beginning of each task
            historical_loader = self.load_historical_data()
            if historical_loader is not None:
                self.cache_important_samples(historical_loader)

        for epoch in range(max_local_epochs):
            # Train on new task data
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

            # Training with cached samples (replay memory)
            # Line 10: Training the local model with cached samples and the new task
            if len(self.cached_samples) > 0 and task > 0:
                self._train_with_cached_samples(epoch)

        if self.args.teval:
            self.grad_eval(old_model=self.model)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _train_with_cached_samples(self, epoch):
        """
        Train with cached important samples for continual learning
        Equation (7) in the paper
        """
        if len(self.cached_samples) == 0:
            return

        # Create a dataloader from cached samples
        cached_x = torch.stack([x for x, y, score in self.cached_samples])
        cached_y = torch.tensor([y for x, y, score in self.cached_samples])

        cached_dataset = TensorDataset(cached_x, cached_y)
        cached_loader = DataLoader(cached_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for x, y in cached_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = self.loss(output, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def load_historical_data(self):
        """
        Load historical data from previous tasks
        Returns a DataLoader with data from all previous tasks
        """
        if not hasattr(self, 'task_dict') or len(self.task_dict) == 0:
            return None

        # Collect all historical data (all tasks except the most recent one)
        historical_data_x = []
        historical_data_y = []

        # Get all task indices except the current task
        task_indices = sorted(self.task_dict.keys())

        if len(task_indices) <= 1:
            # No historical data yet
            return None

        # Collect data from all previous tasks
        for task_idx in task_indices[:-1]:  # Exclude the current task
            task_data = self.task_dict[task_idx]
            if len(task_data) > 0:
                # Assuming task_data is a DataLoader or similar
                for x, y in task_data:
                    historical_data_x.append(x)
                    historical_data_y.append(y)

        if len(historical_data_x) == 0:
            return None

        # Create dataset and dataloader
        historical_x = torch.cat(historical_data_x, dim=0) if len(historical_data_x) > 0 else torch.tensor([])
        historical_y = torch.cat(historical_data_y, dim=0) if len(historical_data_y) > 0 else torch.tensor([])

        historical_dataset = TensorDataset(historical_x, historical_y)
        historical_loader = DataLoader(
            historical_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        return historical_loader

    def set_parameters(self, model):
        """
        Receive global model w^{t-1} from server
        """
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
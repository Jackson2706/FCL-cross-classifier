import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flcore.clients.clientGLFC import GLFCClient, GradientEncodingNet
from flcore.servers.serverbase import Server


class GLFCServer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(GLFCClient)
        
        # Proxy Server Attributes for GLFC
        self.proxy_data = [] # Stores reconstructed images
        self.proxy_labels = []
        self.best_old_model = None
        self.best_acc = -1
        
        # Gradient Encoding Net (Same as clients)
        self.gamma_net = GradientEncodingNet(args.num_classes).to(self.device)
        # self.gamma_net.load_state_dict(torch.load("gamma_net.pth"))

    def reconstruct_prototypes(self, client_gradients):
        """
        Reconstructs images from gradients received from clients.
        This is the 'Deep Leakage' part performed by the Proxy Server.
        [cite: 221, 222]
        """
        reconstructed_images = []
        reconstructed_labels = []
        
        criterion = nn.CrossEntropyLoss()
        
        print("Proxy Server: Reconstructing prototypes from gradients...")
        
        for grads in client_gradients:
            # Initialize dummy data (Gaussian noise)
            dummy_data = torch.randn((1, 3, 32, 32), device=self.device).requires_grad_(True)
            # Infer label from gradient (usually the gradient of the last layer is negative at target class)
            # Simplified: assuming we know the label index corresponding to the grad
            dummy_label = torch.tensor([0], device=self.device) # Placeholder
            
            optimizer = torch.optim.LBFGS([dummy_data])
            
            # Reconstruction Loop [cite: 228, 229]
            for iters in range(10): # Short iterations for demo
                def closure():
                    optimizer.zero_grad()
                    pred = self.gamma_net(dummy_data)
                    dummy_loss = criterion(pred, dummy_label)
                    
                    dummy_dy_dx = torch.autograd.grad(dummy_loss, self.gamma_net.parameters(), create_graph=True)
                    
                    grad_diff = 0
                    for gx, gy in zip(dummy_dy_dx, grads):
                        grad_diff += ((gx - gy) ** 2).sum()
                    
                    grad_diff.backward()
                    return grad_diff
                
                optimizer.step(closure)
            
            reconstructed_images.append(dummy_data.detach())
            reconstructed_labels.append(dummy_label)
            
        return reconstructed_images, reconstructed_labels

    def select_best_model(self, candidates):
        """
        Evaluates historical global models on the reconstructed proxy data
        to find the one that retains old knowledge best.
        """
        if not self.proxy_data:
            return self.global_model

        best_model = None
        best_acc = -1
        
        # Helper to stack data
        data = torch.cat(self.proxy_data)
        labels = torch.cat(self.proxy_labels)
        
        for model in candidates:
            model.eval()
            with torch.no_grad():
                preds = model(data)
                acc = (preds.argmax(1) == labels).float().mean().item()
            
            if acc > best_acc:
                best_acc = acc
                best_model = model
                
        return best_model

    def train(self):
        """
        Main FCIL Training Loop with Proxy Server Logic.
        """
        # History of global models for selection
        model_history = [] 

        for task in range(self.N_TASKS):
            print(f"\n=== Starting Task {task} ===")
            self.current_task = task
            # self._handle_task_transition(task) # Base helper to load data
            
            # 1. Receive Gradients & Reconstruct (First round of new task)
            # In a real system, we'd query clients. Here we simulate receiving from selected clients.
            if task > 0:
                # Simulate receiving gradients from a client detecting transition
                # In real code: call client.generate_prototype_gradients()
                # self.proxy_data = self.reconstruct_prototypes(received_grads)
                pass

            # 2. Select Best Old Model 
            # Proxy server picks best model from history using reconstructed data
            if len(model_history) > 0 and len(self.proxy_data) > 0:
                self.best_old_model = self.select_best_model(model_history)
            else:
                self.best_old_model = copy.deepcopy(self.global_model)

            for round in range(self.global_rounds):
                print(f"Round {round} (Task {task})")
                
                self.selected_clients = self.select_clients()
                
                # Send Global Model AND Best Old Model to clients
                self.send_models() 
                for client in self.selected_clients:
                    # Explicitly set the best old model for L_RD
                    client.best_old_model = copy.deepcopy(self.best_old_model)
                    
                    # Run local training
                    client.train()
                
                self.receive_models()
                self.aggregate_parameters()
                
                # Save current global model to history for future selection
                if round % 5 == 0:
                    model_history.append(copy.deepcopy(self.global_model))
                    
                self.eval(task, round, flag="global")
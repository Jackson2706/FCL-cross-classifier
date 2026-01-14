import torch
import copy
from flcore.servers.serverbase import Server
from flcore.clients.clientFedLwF import FedLwFClient

class FedLwFServer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(FedLwFClient)
        
        # This will store the global model from the previous task
        self.teacher_model = None 

    def train(self):
        """
        FedLwF Main Training Loop.
        """
        for task in range(self.N_TASKS):
            print(f"\n=== FedLwF Task {task} ===")
            self.current_task = task
            # self._handle_task_transition(task) # Helper from previous implementation logic
            
            # 1. Distribute Teacher Model to Clients
            # For Task 0, there is no teacher.
            # For Task > 0, teacher is the model from end of Task t-1.
            
            if task > 0:
                print(f"Distributing Teacher Model (from end of Task {task-1})...")
                for client in self.clients:
                    # We send a deepcopy to ensure clients don't mutate the server's teacher ref
                    client.teacher_model = copy.deepcopy(self.teacher_model)
                    client.current_task = task
            
            # 2. Federated Training Rounds
            for round in range(self.global_rounds):
                self.selected_clients = self.select_clients()
                
                # Send current global model (Student) to clients for training
                self.send_models() 
                
                for client in self.selected_clients:
                    client.train(task=task)
                
                self.receive_models()
                self.aggregate_parameters()
                
                self.eval(task, round, flag="global")
            
            # 3. End of Task: Update Teacher
            # The current Global Model becomes the Teacher for the NEXT task
            print(f"Snapshotting Global Model as Teacher for next task...")
            self.teacher_model = copy.deepcopy(self.global_model)
            
            # Optional: Save to disk to save memory if models are large
            # torch.save(self.teacher_model, f"teacher_task_{task}.pt")
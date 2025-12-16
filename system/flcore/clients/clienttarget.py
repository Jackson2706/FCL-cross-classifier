import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.utils_core.target_utils import *


class clientTARGET(Client):
    def __init__(self, args, id, train_data, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        self.nums = 8000
        self.total_classes = []
        self.syn_data_loader = None
        self.old_network = None
        self.it = None
        self.kd_alpha = 25
        self.synthtic_save_dir = "dataset/synthetic_data"
        
    def train(self, task=None):
        trainloader = self.load_train_data(task=task)
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
                if self.syn_data_loader is not None:
                    syn_inputs = next(iter(self.syn_data_loader)).to(self.device)
                    syn_outputs = self.model(syn_inputs)
                    with torch.no_grad():
                        syn_old_outputs = self.old_network(syn_inputs)
                    kd_loss = KD_loss(syn_outputs, syn_old_outputs, 2)
                    # print("kd_loss: {}".format(kd_loss))
                    loss += self.kd_alpha * kd_loss
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


    def kd_train(self, student, teacher, criterion, optimizer):
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data() 
        data_iter = DataIter(loader)
        for i in range(kd_steps):
            images = data_iter.next().cuda()  
            with torch.no_grad():
                t_out = teacher(images)#["logits"]
            s_out = student(images.detach())#["logits"]
            loss_s = criterion(s_out, t_out.detach())
            optimizer.zero_grad()

            self.fabric.backward(loss_s)
            self.fabric.clip_gradients(student, optimizer, max_norm=1.0, norm_type=2)
            optimizer.step()
        return loss_s.item()
            
    def get_syn_data_loader(self):
        if self.args.dataset =="CIFAR100":
           dataset_size = 50000
        elif self.args.dataset == "IMAGENET1k":
           dataset_size = 1281167
        iters = math.ceil(dataset_size / (self.args.num_clients*self.current_task*self.args.batch_size))
        syn_bs = 16 #int(self.nums/iters)
        data_dir = os.path.join(self.synthtic_save_dir, "task_{}".format(self.current_task-1))
        print("iters{}, syn_bs:{}, data_dir: {}".format(iters, syn_bs, data_dir))

        syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.nums)
        syn_data_loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=syn_bs, shuffle=True,
            num_workers=0)
        return syn_data_loader

    def get_all_syn_data(self):
        data_dir = os.path.join(self.synthtic_save_dir, "task_{}".format(self.current_task))
        syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.nums)
        loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=sample_batch_size, shuffle=True,
            num_workers=0, pin_memory=True, sampler=None)
        return loader

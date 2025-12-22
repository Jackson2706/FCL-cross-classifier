import time
import torch
import copy
from flcore.clients.clienttarget import clientTARGET
from flcore.servers.serverbase import Server
from flcore.utils_core.target_utils import *
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
from utils.model_utils import ParamDict
from torch.nn.utils import vector_to_parameters, parameters_to_vector

from torch.optim.lr_scheduler import StepLR
import numpy as np

import statistics

from torch import nn
from torch.nn import functional as F
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from typing import List
from torch.func import functional_call
from copy import deepcopy

from tqdm import tqdm
from torchvision import transforms
import time, os, math
import torch.nn.init as init
from PIL import Image
from torch.autograd import Variable
from abc import ABC
import shutil

class FedTARGET(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.synthtic_save_dir = "dataset/synthetic_data"
        if os.path.exists(self.synthtic_save_dir):
            shutil.rmtree(self.synthtic_save_dir)
        self.nums = 8000
        self.total_classes = []
        self.syn_data_loader = None
        self.old_network = None
        self.kd_alpha = 25
        if "CIFAR100" in self.dataset:
            self.dataset_size = 50000
        elif "IMAGENET1k" in self.dataset:
            self.dataset_size = 1281167
        self.available_labels_current = None

        self.set_clients(clientTARGET)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):

        if self.args.num_tasks % self.N_TASKS != 0:
            raise ValueError("Set num_task again")

        for task in range(self.args.num_tasks):

            print(f"\n================ Current Task: {task} =================")
            if task == 0:
                 # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                self.available_labels_current = available_labels_current
                
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            else:
                self.current_task = task
                
                torch.cuda.empty_cache()

                self.old_network = deepcopy(self.global_model)
                for p in self.old_network.parameters():
                    p.requires_grad = False

                for i in range(len(self.clients)):
                    
                    if self.args.dataset == 'IMAGENET1k':
                        train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    elif self.args.dataset == 'CIFAR100':
                        train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    else:
                        raise NotImplementedError("Not supported dataset")

                    # update dataset
                    self.clients[i].next_task(train_data, label_info)
                    self.clients[i].old_network = deepcopy(self.clients[i].model)
                    for p in self.clients[i].old_network.parameters():
                        p.requires_grad = False
                    
                # update labels info.
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                self.available_labels_current = available_labels_current

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            # ============ train ==============

            for i in range(self.global_rounds):
                
                if task > 0 :
                    for u in self.clients:
                        u.old_network = deepcopy(self.old_network)
                        u.syn_data_loader = u.get_syn_data_loader()
                        u.it = DataIter(u.syn_data_loader)
                        u.old_network = u.old_network.to(self.device)

                glob_iter = i + self.global_rounds * task
                s_t = time.time()
                self.selected_clients = self.select_clients()
                self.send_models()

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                for client in self.selected_clients:
                    client.train(task=task)

                self.receive_models()
                self.receive_grads()
                model_origin = copy.deepcopy(self.global_model)
                self.aggregate_parameters()

                if self.args.seval:
                    self.spatio_grad_eval(model_origin=model_origin)

                if self.args.pca_eval:
                    self.proto_eval(model=self.global_model, task=task, round=i)

                if i%self.eval_gap == 0:
                    self.eval(task=task, glob_iter=glob_iter, flag="local")

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            self.data_generation(task=task, available_labels_current=self.available_labels_current)

            if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
                if self.args.offlog == True and not self.args.debug:  
                    self.eval_task(task=task, glob_iter=glob_iter, flag="local")

                    # need eval before data update
                    self.send_models()
                    self.eval_task(task=task, glob_iter=glob_iter, flag="global")

    def kd_train(self, student, teacher, criterion, optimizer, task):
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data(task=task) 
        data_iter = DataIter(loader)
        for i in range(kd_steps):
            images = data_iter.next().to(self.device)
            with torch.no_grad():
                t_out = teacher(images)#["logits"]
            s_out = student(images.detach())#["logits"]
            loss_s = criterion(s_out, t_out.detach())
            optimizer.zero_grad()

            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
        return loss_s.item()


    def data_generation(self, task, available_labels_current):
        nz = 256
        #img_size = 32 if dataset == "cifar100" else 64
        #if dataset == "imagenet100": img_size = 128 
        #    
        #img_shape = (3, 32, 32) if dataset == "cifar100" else (3, 64, 64)
        #if dataset == "imagenet100": img_shape = (3, 128, 128) #(3, 224, 224)
        # img_size = 224
        # img_shape = (3, 224, 224)
        img_size = 32
        img_shape = (3, 32, 32)
        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=3, device=self.device).to(self.device)
        student = deepcopy(self.global_model)
        student.apply(weight_init)
        tmp_dir = os.path.join(self.synthtic_save_dir, "task_{}".format(task))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir) 
        synthesizer = GlobalSynthesizer(deepcopy(self.global_model), student, generator,
                    nz=nz, allowed_classes=available_labels_current, img_size=img_shape, init_dataset=None,
                    save_dir=tmp_dir,
                    transform=train_transform, normalizer=normalizer,
                    synthesis_batch_size=synthesis_batch_size, sample_batch_size=sample_batch_size,
                    iterations=g_steps, warmup=warmup, lr_g=lr_g, lr_z=lr_z,
                    adv=adv, bn=bn, oh=oh,
                    reset_l0=reset_l0, reset_bn=reset_bn,
                    bn_mmt=bn_mmt, is_maml=is_maml, fabric=None, args = self.args)#, args=self.args)
        
        criterion = KLDiv(T=T)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.002, weight_decay=0.0001,
                            momentum=0.9)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=2e-4)

        for it in tqdm(range(syn_round), desc="Data Generation"):
            synthesizer.synthesize() # generate synthetic data
            if it >= warmup:
                loss = self.kd_train(student, self.global_model, criterion, optimizer, task) # kd_steps
                #test_acc = self._compute_accuracy(student, self.test_loader)
                #print("Task {}, Data Generation, Epoch {}/{} =>  Student test_acc: {:.2f}".format(
                #    self.cur_task, it + 1, syn_round, test_acc,))
                print("Task {}, Data Generation, Epoch {}/{} =>  Student loss: {:.2f}".format(
                    task, it + 1, syn_round, loss,))
                #scheduler.step()
                # wandb.log({'Distill {}, accuracy'.format(self.cur_task): test_acc})


        print("For task {}, data generation completed! ".format(task))  

            
    # def get_syn_data_loader(self):
    #     #if self.args["dataset"] =="cifar100":
    #     #    dataset_size = 50000
    #     #elif self.args["dataset"] == "tiny_imagenet":
    #     #    dataset_size = 100000
    #     #elif self.args["dataset"] == "imagenet100":
    #     #    dataset_size = 130000
    #     dataset_size = self.dataset_size
    #     iters = math.ceil(dataset_size / (self.num_clients*self.N_TASKS*self.batch_size))
    #     syn_bs = 32 #int(self.nums/iters)
    #     data_dir = os.path.join(self.save_dir, "task_{}".format(self.cur_task-1))
    #     print("iters{}, syn_bs:{}, data_dir: {}".format(iters, syn_bs, data_dir))

    #     syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.nums)
    #     syn_data_loader = torch.utils.data.DataLoader(
    #         syn_dataset, batch_size=syn_bs, shuffle=True,
    #         num_workers=0)
    #     return syn_data_loader

    def get_all_syn_data(self, task):
        data_dir = os.path.join(self.synthtic_save_dir, "task_{}".format(task))
        syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.nums)
        loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=sample_batch_size, shuffle=True,
            num_workers=0, pin_memory=True, sampler=None)
        return loader
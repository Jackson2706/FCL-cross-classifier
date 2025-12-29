import os
import sys
import copy
import torch
import argparse
import time
import warnings
import numpy as np
import torchvision
import json
import wandb
from argparse import Namespace
from types import SimpleNamespace

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverala import FedALA
from flcore.servers.serverdbe import FedDBE
from flcore.servers.serveras import FedAS
from flcore.servers.serverweit import FedWeIT
from flcore.servers.serveraffcl import FedAFFCL
from flcore.servers.servertarget import FedTARGET
from flcore.servers.serverl2p import FedL2P
from flcore.servers.serverssi import FedSSI
from flcore.servers.serverrefedplus import ReFedPlus
from flcore.servers.ours_v2 import OursV2
from flcore.servers.ours import Ours
from flcore.servers.servercoplay import serverCoplay
from flcore.servers.serverdgr import
from flcore.trainmodel.models import *

from flcore.trainmodel.AFFCL_models import AFFCLModel
from flcore.servers.serverstgm import FedSTGM
from flcore.servers.serverfcil import FedFCIL

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *
from flcore.trainmodel.vit_prompt_l2p import *

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    
    if args.wandb:
        wandb.login(key="22e152c33a1e551f449b7a2f029bf1258f6e8367")
        wandb.init(
            project="FCL",
            entity="jackson2706",
            config=args, 
            name=f"{args.dataset}_{args.model}_{args.algorithm}_{args.optimizer}_lr{args.local_learning_rate}_{args.note}" if args.note else f"{args.dataset}_{args.model}_{args.algorithm}_{args.optimizer}_lr{args.local_learning_rate}", 
        )

    time_list = []
    model_str = args.model
    args.model_str = model_str

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "CNN": # non-convex
            if "CIFAR100" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "CIFAR10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "IMAGENET1k" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "IMAGENET1k224" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=179776).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "ResNet50":
            args.model = torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "ResNet50-pretrained":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            args.model = torchvision.models.resnet50(weights=weights, num_classes=args.num_classes).to(args.device)
        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "Swin_t":
            args.model = torchvision.models.swin_t(weights=None, num_classes=args.num_classes).to(args.device)
        elif model_str == "AFFCLModel":    
            args.model = AFFCLModel(args).to(args.device)
        elif model_str == "VitL2P":
            args.model = VitL2P(
                num_classes=args.num_classes,
                n_prompts=args.n_prompts,
                prompt_length=args.prompt_length,
                prompt_pool=args.prompt_pool,
                pool_size=args.pool_size,
            ).to(args.device)
        else:
            raise NotImplementedError

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "FedSSI":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedSSI(args, i)

        elif args.algorithm == "ReFedPlus":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = ReFedPlus(args, i)

        elif args.algorithm == "FedALA":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedALA(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)

        elif args.algorithm == "FedWeIT":
            # args.model = None
            server = FedWeIT(args, i)

        elif args.algorithm == "PreciseFCL":
            # args.head = copy.deepcopy(args.model.classifier.fc_classifier)
            # args.model.classifier.fc_classifier = nn.Identity()
            # args.model.classifier = BaseHeadSplit_AFFCL(args.model.classifier, args.head)
            server = FedAFFCL(args, i)

        elif args.algorithm == 'FedAS':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)

        elif args.algorithm == "FedFCIL":
            server = FedFCIL(args, i)
            
        elif args.algorithm == "FedSTGM":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedSTGM(args, i)

        elif args.algorithm == "FedTARGET":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedTARGET(args, i)

        elif args.algorithm == "FedL2P":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedL2P(args, i)
        elif args.algorithm == "Ours":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = Ours(args, i)
        elif args.algorithm == "Ours_v2":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = OursV2(args, i)
        elif args.algorithm == "Coplay":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = serverCoplay(args, i)
        elif args.algorithm == "FedDGR":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = serverFedDGR(args, i)
        elif args.algorithm == "FedCIL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = serverCoplay(args, i)
        else:
            print(args.algorithm)
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    # Global average
    print("All done!")

if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfp', type=str, default=None, help='Configuration path for training')
    parser.add_argument('--note', type=str, default=None, help='Optional note to add to save name')
    parser.add_argument('--wandb', type=bool, default=False, help='Log on wandb')
    parser.add_argument('--offlog', type=bool, default=False, help='Save wandb logger')
    parser.add_argument('--log', type=bool, default=False, help='Print logger')
    parser.add_argument('--debug', type=bool, default=False, help='When use Debug, turn off forgetting')
    parser.add_argument('--cpt', type=int, default=20, help='Class per task')
    parser.add_argument('--nt', type=int, default=None, help='Num tasks')
    parser.add_argument('--seval', action='store_true', help='Log Spatio Gradient')
    parser.add_argument('--teval', action='store_true', help='Log Temporal Gradient')
    parser.add_argument('--pca_eval', action='store_true', help='Log PCA Gradient')

    args = parser.parse_args()

    with open(args.cfp, 'r') as f:
        cfdct = json.load(f)
    if args.note is not None:
        cfdct['note'] = args.note
    if args.nt is not None:
        cfdct['num_tasks'] = args.nt


    cfdct['nt'] = args.nt
    cfdct['wandb'] = args.wandb
    cfdct['offlog'] = args.offlog
    cfdct['log'] = args.log
    cfdct['debug'] = args.debug
    cfdct['cpt'] = args.cpt
    cfdct['seval'] = args.seval
    cfdct['teval'] = args.teval
    cfdct['pca_eval'] = args.pca_eval

    print(args.seval)
    print(args.teval)
    print(args.pca_eval)

    if "tgm" not in cfdct:
        cfdct['tgm'] = True

    if "coreset" not in cfdct:
        cfdct['coreset'] = False

    args = Namespace(**cfdct)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    print(args)
    run(args)


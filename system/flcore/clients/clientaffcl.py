import torch
import torch.nn as nn
import glog as logger
import numpy as np
import wandb
import copy

from flcore.clients.clientbase import Client
from flcore.utils_core.AFFCL_utils import str_in_list, Meter

eps = 1e-30

class ClientAFFCL(Client):
    def __init__(self, args, id, train_data, classifier_head_list=[], **kwargs):
        super().__init__(args, id, train_data, **kwargs)
        
        self.args = args
        self.k_loss_flow = args.k_loss_flow
        self.classifier_head_list = classifier_head_list
        self.use_lastflow_x = args.use_lastflow_x
        self.classifier_global_mode = args.classifier_global_mode
        self.beta = args.beta
        self.local_model_name = copy.deepcopy(list(self.model.named_parameters()))
        self.init_loss_fn()
    
    def train(
        self,
        task,
        glob_iter,
        global_classifier,
        verbose
    ):
        '''
        @ glob_iter: the overall iterations across all tasks
        
        '''
        trainloader = self.load_train_data(task=task)
        correct = 0
        sample_num = 0
        cls_meter = Meter()
        for iteration in range(self.local_epochs):
            counter = 0
            for i, (x, y) in enumerate(trainloader):
                if counter >= 1:
                    break
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                last_classifier = None
                last_flow = None
                if type(self.last_copy)!=type(None):
                    last_classifier = self.last_copy.classifier
                    last_classifier.eval()
                    if self.algorithm=='PreciseFCL':
                        last_flow = self.last_copy.flow
                        last_flow.eval()

                if self.algorithm=='PreciseFCL' and self.k_loss_flow>0:
                    self.model.classifier.eval()
                    self.model.flow.train()
                    # print("Classifier device:", next(self.model.classifier.parameters()).device)
                    # print("Flow device:", next(self.model.flow.parameters()).device)
                    # self.model.flow.to(self.device)
                    flow_result = self.model.train_a_batch(
                        x, y, train_flow=True, flow=None, last_flow=last_flow,
                        last_classifier = last_classifier,
                        global_classifier = global_classifier,
                        classes_so_far = self.classes_so_far,
                        classes_past_task = self.classes_past_task,
                        available_labels = self.available_labels,
                        available_labels_past = self.available_labels_past)
                    cls_meter._update(flow_result, batch_size=x.shape[0])

                flow = None
                if self.algorithm=='PreciseFCL':
                    if self.use_lastflow_x:
                        flow = last_flow
                    else:
                        flow = self.model.flow
                        flow.eval()
                        # print("Classifier device:", next(self.model.classifier.parameters()).device)
                        # print("Flow device:", next(self.model.flow.parameters()).device)

                self.model.classifier.train()
                # print("Classifier device:", next(self.model.classifier.parameters()).device)
                # print("Flow device:", next(self.model.flow.parameters()).device)
                cls_result = self.model.train_a_batch(
                    x, y, train_flow=False, flow=flow, last_flow=last_flow,
                    last_classifier = last_classifier,
                    global_classifier = global_classifier,
                    classes_so_far = self.classes_so_far,
                    classes_past_task = self.classes_past_task,
                    available_labels = self.available_labels,
                    available_labels_past = self.available_labels_past)

                #c_loss_all += result['c_loss']
                correct += cls_result['correct']
                sample_num += x.shape[0]
                cls_meter._update(cls_result, batch_size=x.shape[0])
                counter += 1

        if self.args.teval:
            self.grad_eval(old_model=self.model.classifier)

        acc = float(correct)/sample_num
        result_dict = cls_meter.get_scalar_dict('global_avg')
        if 'flow_loss' not in result_dict.keys():
            result_dict['flow_loss'] = 0
        if 'flow_loss_last' not in result_dict.keys():
            result_dict['flow_loss_last'] = 0

        if verbose:
            logger.info(("Training for user {:d}; Acc: {:.2f} %%; c_loss: {:.4f}; kd_loss: {:.4f}; flow_prob_mean: {:.4f}; "
                         "flow_loss: {:.4f}; flow_loss_last: {:.4f}; c_loss_flow: {:.4f}; kd_loss_flow: {:.4f}; "
                         "kd_loss_feature: {:.4f}; kd_loss_output: {:.4f}").format(
                                        self.id, acc*100.0, result_dict['c_loss'], result_dict['kd_loss'],
                                        result_dict['flow_prob_mean'], result_dict['flow_loss'], result_dict['flow_loss_last'],
                                        result_dict['c_loss_flow'], result_dict['kd_loss_flow'],
                                        result_dict['kd_loss_feature'], result_dict['kd_loss_output']))

        return {'acc': acc, 'c_loss': result_dict['c_loss'], 'kd_loss': result_dict['kd_loss'], 'flow_prob_mean': result_dict['flow_prob_mean'],
                 'flow_loss': result_dict['flow_loss'], 'flow_loss_last': result_dict['flow_loss_last'], 'c_loss_flow': result_dict['c_loss_flow'],
                   'kd_loss_flow': result_dict['kd_loss_flow']}

    def test_metrics(self, task):
        testloader = self.load_test_data(task=task)

        self.model.classifier.eval()
        self.model.flow.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output, _, _ = self.model.classifier(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        
        return test_acc, test_num

    def train_metrics(self, task):
        trainloader = self.load_train_data(task=task)
        self.model.classifier.eval()
        self.model.flow.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output, _, _ = self.model.classifier(x)
                loss = self.model.classify_criterion(torch.log(output+1e-30), y)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
    
    def init_loss_fn(self):
        self.loss=nn.NLLLoss()
        self.dist_loss = nn.MSELoss()
        self.ensemble_loss=nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def set_parameters_precise(self, model, beta=1):
        '''
        self.model: old user model
        model: the global model on the server (new model)
        '''
        for (name1, old_param), (name2, new_param), (name3, local_param) in zip(
                self.model.named_parameters(), model.named_parameters(), self.local_model_name):
            assert name1==name2==name3
            if (self.algorithm=='PreciseFCL') and (self.classifier_global_mode=='head') and \
                    ('classifier' in name1) and (not str_in_list(name1, self.classifier_head_list)):
                continue
            elif (self.algorithm=='PreciseFCL') and (self.classifier_global_mode=='extractor') and \
                    ('classifier' in name1) and (str_in_list(name1, self.classifier_head_list)):
                continue
            elif (self.algorithm=='PreciseFCL') and (self.classifier_global_mode=='none') and 'classifier' in name1:
                continue
            else:
                if beta == 1:
                    old_param.data = new_param.data.clone()
                    local_param.data = new_param.data.clone()
                else:
                    old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                    local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()

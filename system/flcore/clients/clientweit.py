import math
import torch
import numpy as np
import time
import copy

from flcore.clients.clientbase import Client
from flcore.trainmodel.fedewit_models import *
from flcore.utils_core.fedweit_utils import *

class clientWeIT(Client):
    def __init__(self, args, id, train_data, initial_weights, **kwargs):
        super().__init__(args, id, train_data, **kwargs)

        self.args = args
        self.device = self.args.device
        self.state = {'gpu_id': self.device}
        self.logger = Logger(self.args)

        self.nets = NetModule(self.args)
        self.train = TrainModule(self.args, self.logger, self.nets)
        self.trainloader = self.load_train_data(task=0)

        self.init_model(initial_weights)

    def init_model(self, initial_weights):
        decomposed = True if self.args.algorithm in ['FedWeIT'] else False
        if self.args.base_network == 'lenet':
            self.nets.build_lenet(initial_weights, decomposed=decomposed)

    def switch_state(self, client_id):
        if self.is_new(client_id):
            self.nets.init_state(client_id)
            self.train.init_state(client_id)
            self.init_state(client_id)
            self.load_data()
        else: # load_state
            self.load_state(client_id)
            self.nets.load_state(client_id)
            self.train.load_state(client_id)
            self.load_data()

    def is_new(self, client_id):
        return not os.path.exists(os.path.join(self.args.state_dir, f'{client_id}_client.npy'))

    def init_state(self, cid):
        self.state['client_id'] = cid
        self.state['task_names'] = {}
        self.state['curr_task'] =  -1
        self.state['round_cnt'] = 0
        self.state['done'] = False

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_client.npy'.format(cid))).item()
        self.update_train_config_by_tid(self.state['curr_task'])

    def save_state(self):
        np_save(self.args.state_dir, '{}_client.npy'.format(self.state['client_id']), self.state)
        self.nets.save_state()
        self.train.save_state()

    def init_new_task(self):
        # print("hi")
        self.state['curr_task'] += 1
        self.state['round_cnt'] = 0
        self.train.init_learning_rate()
        self.update_train_config_by_tid(self.state['curr_task'])
        self.load_data()

    def update_train_config_by_tid(self, tid):
        self.target_model = self.nets.get_model_by_tid(tid)
        self.trainable_variables = self.nets.get_trainable_variables(tid)
        self.trainable_variables_body = self.nets.get_trainable_variables(tid, head=False)
        self.train.set_details({
            'loss': self.loss,
            'model': self.target_model,
            'trainables': self.trainable_variables,
        })

    def load_data(self):
        self.train.set_task({
            'trainloader': self.trainloader,
        })

    def get_model_by_tid(self, tid):
        return self.nets.get_model_by_tid(tid)

    def set_weights(self, weights):
        if self.args.algorithm in ['FedWeIT']:
            if weights is None:
                return None
            for i, w in enumerate(weights):
                w = torch.tensor(w) if isinstance(w, np.ndarray) else w
                sw = self.nets.get_variable('shared', i).detach().cpu()
                residuals = torch.eq(w, torch.zeros_like(w)).float()

                # print(w.device)
                # print(sw.device)
                # print(residuals.device)
                
                sw.data = sw * residuals + w
                sw.data = sw.data.to(self.device)
        else:
            self.nets.set_body_weights(weights)

    def get_weights(self):
        if self.args.algorithm in ['FedWeIT']:
            if self.args.sparse_comm:
                hard_threshold = []
                sw_pruned = []
                masks = self.nets.decomposed_variables['mask'][self.state['curr_task']]
                for lid, sw in enumerate(self.nets.decomposed_variables['shared']):
                    mask = masks[lid]
                    m_sorted = torch.sort(torch.flatten(torch.abs(mask)))[0]
                    thres = m_sorted[math.floor(len(m_sorted) * (self.args.client_sparsity))]
                    m_binary = torch.gt(torch.abs(mask), thres).float().cpu().numpy().tolist()
                    hard_threshold.append(m_binary)

                    m_binary = np.array(m_binary)
                    num_dims = sw.dim()

                    for _ in range(1, num_dims):  
                        m_binary = np.expand_dims(m_binary, axis=-1)
                    # print("numdim " + str(num_dims))
                    # print("swshape" + str(sw.detach().cpu().numpy().shape))

                    # print(m_binary.shape)
                    
                    sw_pruned.append(sw.detach().cpu().numpy() * m_binary)
                # self.train.calculate_communication_costs(sw_pruned)
                return sw_pruned, hard_threshold
            else:
                return [sw.detach().cpu().numpy() for sw in self.nets.decomposed_variables['shared']]
        else:
            return self.nets.get_body_weights()

    def get_train_size(self):
        return self.args.batch_size

    def get_task_id(self):
        return self.curr_task

    def stop(self):
        self.done = True

    def train_one_round(self, client_id, curr_round, selected, global_weights=None, from_kb=None):
        ######################################
        self.switch_state(client_id)
        ######################################
        self.state['round_cnt'] += 1
        self.state['curr_round'] = curr_round
        
        if from_kb is not None:
            for lid, weights in enumerate(from_kb):
                tid = self.state['curr_task'] 
                self.nets.decomposed_variables['from_kb'][tid][lid].data = torch.tensor(weights)
        
        if self.state['curr_task'] < 0:
            self.init_new_task()
            self.set_weights(global_weights) 
        else:
            is_last_task = (self.state['curr_task'] == self.args.num_tasks - 1)
            is_last_round = (self.state['round_cnt'] % self.args.global_rounds == 0 and self.state['round_cnt'] != 0)
            is_last = is_last_task and is_last_round
            if is_last_round:
                if is_last_task:
                    if self.train.state['early_stop']:
                        self.train.evaluate()
                    self.stop()
                    return
                else:
                    self.init_new_task()
                    self.state['prev_body_weights'] = self.nets.get_body_weights(self.state['curr_task'])

        if selected:
            self.set_weights(global_weights)

        self.train.train_one_round(self.state['curr_round'], self.state['round_cnt'], self.state['curr_task'])
        
        # self.logger.save_current_state(self.state['client_id'], {
        #     'scores': self.train.get_scores(),
        #     'capacity': self.train.get_capacity(),
        #     'communication': self.train.get_communication()
        # })
        self.save_state()
        
        if selected:
            return self.get_weights(), self.get_train_size()

    def loss(self, y_true, y_pred):
        weight_decay, sparseness, approx_loss = 0, 0, 0
        loss = torch.nn.functional.cross_entropy(y_pred, y_true)
        for lid in range(len(self.nets.shapes)):
            sw = self.nets.get_variable(var_type='shared', lid=lid)
            aw = self.nets.get_variable(var_type='adaptive', lid=lid, tid=self.state['curr_task'])
            mask = self.nets.get_variable(var_type='mask', lid=lid, tid=self.state['curr_task'])
            g_mask = self.nets.generate_mask(mask)
            weight_decay += self.args.wd * torch.nn.functional.mse_loss(aw, torch.zeros_like(aw))
            weight_decay += self.args.wd * torch.nn.functional.mse_loss(mask, torch.zeros_like(mask))
            sparseness += self.args.lambda_l1 * torch.sum(torch.abs(aw))
            sparseness += self.args.lambda_mask * torch.sum(torch.abs(mask))
            if self.state['curr_task'] == 0:
                weight_decay += self.args.wd * torch.nn.functional.mse_loss(sw, torch.zeros_like(sw))
            else:
                for tid in range(self.state['curr_task']):
                    prev_aw = self.nets.get_variable(var_type='adaptive', lid=lid, tid=tid)
                    prev_mask = self.nets.get_variable(var_type='mask', lid=lid, tid=tid)
                    g_prev_mask = self.nets.generate_mask(prev_mask)
                    #################################################
                    restored = sw * g_prev_mask + prev_aw
                    a_l2 = torch.nn.functional.mse_loss(restored, torch.tensor(self.state['prev_body_weights'][lid][tid]))
                    approx_loss += self.args.lambda_l2 * a_l2
                    #################################################
                    sparseness += self.args.lambda_l1 * torch.sum(torch.abs(prev_aw))
        
        loss += weight_decay + sparseness + approx_loss 
        return loss

    def get_adaptives(self):
        adapts = []
        for lid in range(len(self.nets.shapes)):
            aw = self.nets.get_variable(var_type='adaptive', lid=lid, tid=self.state['curr_task']).detach().cpu().numpy()
            hard_threshold = np.greater(np.abs(aw), self.args.lambda_l1).astype(np.float32)
            adapts.append(aw * hard_threshold)
        return adapts

    def test_metrics(self, task):
        testloader = self.load_test_data(task=task)
        if hasattr(clientWeIT, 'state'):
            self.curr_model = self.nets.get_model_by_tid(self.state['curr_task'])
        else:
            self.curr_model = self.nets.get_model_by_tid(0)
        # self.model = self.load_model('model')
        self.curr_model.to(self.device)
        self.curr_model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.curr_model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        
        return test_acc, test_num

    def train_metrics(self, task):
        trainloader = self.load_train_data(task=task)
        if hasattr(clientWeIT, 'state'):
            self.curr_model = self.nets.get_model_by_tid(self.state['curr_task'])
        else:
            self.curr_model = self.nets.get_model_by_tid(0)
        # self.model = self.load_model('model')
        self.curr_model.to(self.device)
        self.curr_model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.curr_model.eval()(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    def next_task(self, train, label_info = None, if_label = True):
        
        # if self.args.algorithm != "PreciseFCL" and self.learning_rate_decay:
        #     # update last model:
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = self.learning_rate  # Đặt lại về giá trị ban đầu

        #     self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #         optimizer=self.optimizer, 
        #         gamma=self.args.learning_rate_decay_gamma
        #     )

        # self.last_copy  = copy.deepcopy(self.model)
        # self.last_copy.cuda()
        self.if_last_copy = True
        
        # update dataset: 
        self.train_data = train
        self.train_targets = [label for _, label in self.train_data]
        
        self.classes_past_task = copy.deepcopy(self.classes_so_far)
        self.current_task += 1
        self.trainloader = self.load_train_data(task=self.current_task)

        # update classes_so_far
        if if_label:
            self.classes_so_far.extend(label_info['labels'])
            self.task_dict[self.current_task] = label_info['labels']

            self.current_labels.clear()
            self.current_labels.extend(label_info['labels'])
            
        return

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import torch.nn.init as init
import math
import psutil
import os

from flcore.utils_core.fedweit_utils import *

class TrainModule:

    def __init__(self, args, logger, nets):
        self.args = args
        # self.logger = logger
        self.nets = nets
        self.device = args.device

    def init_state(self, cid):
        self.state = {
            'client_id': cid,
            'scores': {
                'test_loss': {},
                'test_acc': {},
            },
            'capacity': {
                'ratio': [],
                'num_shared_activ': [],
                'num_adapts_activ': [],
            },
            'communication': {
                'ratio': [],
                'num_actives': [],
            },
            'num_total_params': 0,
            'optimizer': []
        }
        self.init_learning_rate()

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_train.npy'.format(cid))).item()
        self.optimizer = optim.Adam(list(self.params['trainables']))
        self.optimizer.load_state_dict(self.state['optimizer'])

    def save_state(self):
        self.state['optimizer'] = self.optimizer.state_dict()
        np_save(self.args.state_dir, '{}_train.npy'.format(self.state['client_id']), self.state)

    def init_learning_rate(self):
        self.state['early_stop'] = False
        self.state['lowest_lss'] = np.inf
        self.state['curr_lr'] = self.args.local_learning_rate
        self.state['curr_lr_patience'] = self.args.lr_patience
        self.init_optimizer(self.state['curr_lr'])

    def init_optimizer(self, curr_lr):
        self.optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=curr_lr)

    # def adaptive_lr_decay(self):
    #     vlss = self.vlss
    #     if vlss < self.state['lowest_lss']:
    #         self.state['lowest_lss'] = vlss
    #         self.state['curr_lr_patience'] = self.args.lr_patience
    #     else:
    #         self.state['curr_lr_patience'] -= 1
    #         if self.state['curr_lr_patience'] <= 0:
    #             prev = self.state['curr_lr']
    #             self.state['curr_lr'] /= self.args.lr_factor
    #             self.logger.print(self.state['client_id'], 'epoch:%d, learning rate has been dropped from %.5f to %.5f' \
    #                                                 %(self.state['curr_epoch'], prev, self.state['curr_lr']))
    #             if self.state['curr_lr'] < self.args.lr_min:
    #                 self.logger.print(self.state['client_id'], 'epoch:%d, early-stopped as minimum lr reached to %.5f'%(self.state['curr_epoch'], self.state['curr_lr']))
    #                 self.state['early_stop'] = True
    #             self.state['curr_lr_patience'] = self.args.lr_patience
    #             for param_group in self.optimizer.param_groups:
    #                 param_group['lr'] = self.state['curr_lr']

    def train_one_round(self, curr_round, round_cnt, curr_task):
        self.state['curr_round'] = curr_round
        self.state['round_cnt'] = round_cnt
        self.state['curr_task'] = curr_task

        trainloader = self.task['trainloader']
        self.curr_model = self.nets.get_model_by_tid(curr_task)
        self.curr_model.to(self.device)

        self.params['trainables'] = [param.to(self.device) for param in self.params['trainables']]
        self.optimizer = torch.optim.Adam(self.params['trainables'], lr=self.state['curr_lr'])

        # for param in self.params['trainables']:
        #     print("param: " + str(param.device))
        # for param in self.curr_model.parameters():
        #     print("model: " + str(param.device))
        # print("model: " + str(param.device))
        # print("param: " + str(param.device))

        self.curr_model.train()
        for epoch in range(self.args.local_epochs):
            self.state['curr_epoch'] = epoch + 1
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # if self.train_slow:
                #     time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.curr_model(x)
                loss = self.params['loss'](output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # self.validate()
            # self.evaluate()
            # if self.args.algorithm in ['FedWeIT']:
            #     self.calculate_capacity()
            # self.adaptive_lr_decay()
            if self.state['early_stop']:
                continue

    # def validate(self):
    #     self.curr_model.eval()
    #     with torch.no_grad():
    #         for i in range(0, len(self.task['x_valid']), self.args.batch_size):
    #             x_batch = torch.tensor(self.task['x_valid'][i:i+self.args.batch_size])
    #             y_batch = torch.tensor(self.task['y_valid'][i:i+self.args.batch_size])
    #             y_pred = self.curr_model(x_batch)
    #             loss = nn.functional.cross_entropy(y_pred, y_batch)
    #             self.add_performance('valid_lss', 'valid_acc', loss, y_batch, y_pred)
    #     self.vlss, self.vacc = self.measure_performance('valid_lss', 'valid_acc')

    def evaluate(self):
        self.curr_model.eval()
        with torch.no_grad():
            for tid in range(self.state['curr_task'] + 1):
                if self.args.model == 'stl':
                    if not tid == self.state['curr_task']:
                        continue
                x_test = torch.tensor(self.task['x_test_list'][tid])
                y_test = torch.tensor(self.task['y_test_list'][tid])
                model = self.nets.get_model_by_tid(tid)
                for i in range(0, len(x_test), self.args.batch_size):
                    x_batch = x_test[i:i+self.args.batch_size]
                    y_batch = y_test[i:i+self.args.batch_size]
                    y_pred = model(x_batch)
                    loss = nn.functional.cross_entropy(y_pred, y_batch)
                    self.add_performance('test_lss', 'test_acc', loss, y_batch, y_pred)
                lss, acc = self.measure_performance('test_lss', 'test_acc')
                if not tid in self.state['scores']['test_loss']:
                    self.state['scores']['test_loss'][tid] = []
                    self.state['scores']['test_acc'][tid] = []
                self.state['scores']['test_loss'][tid].append(lss)
                self.state['scores']['test_acc'][tid].append(acc)
                self.logger.print(self.state['client_id'], 'round:{}(cnt:{}),epoch:{},task:{},lss:{},acc:{} ({},#_train:{},#_valid:{},#_test:{})'
                    .format(self.state['curr_round'], self.state['round_cnt'], self.state['curr_epoch'], tid, round(lss, 3), \
                        round(acc, 3), len(self.task['x_train']), len(self.task['x_valid']), len(x_test)))

    # def add_performance(self, lss_name, acc_name, loss, y_true, y_pred):
    #     self.metrics[lss_name].update(loss)
    #     self.metrics[acc_name].update(y_pred, y_true)

    # def measure_performance(self, lss_name, acc_name):
    #     lss = float(self.metrics[lss_name].compute())
    #     acc = float(self.metrics[acc_name].compute())
    #     self.metrics[lss_name].reset()
    #     self.metrics[acc_name].reset()
    #     return lss, acc

    # def calculate_capacity(self):
    #     def l1_pruning(weights, hyp):
    #         hard_threshold = torch.gt(torch.abs(weights), hyp).float()
    #         return weights * hard_threshold

    #     if self.state['num_total_params'] == 0:
    #         for dims in self.nets.shapes:
    #             params = 1
    #             for d in dims:
    #                 params *= d
    #             self.state['num_total_params'] += params
    #     num_total_activ = 0
    #     num_shared_activ = 0
    #     num_adapts_activ = 0
    #     for var_name in self.nets.decomposed_variables:
    #         if var_name == 'adaptive':
    #             for tid in range(self.state['curr_task'] + 1):
    #                 for lid in self.nets.decomposed_variables[var_name][tid]:
    #                     var = self.nets.decomposed_variables[var_name][tid][lid]
    #                     var = l1_pruning(var, self.args.lambda_l1)
    #                     actives = torch.ne(var, torch.zeros_like(var)).float()
    #                     actives = torch.sum(actives)
    #                     num_adapts_activ += actives
    #         elif var_name == 'shared':
    #             for var in self.nets.decomposed_variables[var_name]:
    #                 actives = torch.ne(var, torch.zeros_like(var)).float()
    #                 actives = torch.sum(actives)
    #                 num_shared_activ += actives
    #         else:
    #             continue
    #     num_total_activ += (num_adapts_activ + num_shared_activ)
    #     ratio = num_total_activ / self.state['num_total_params']
    #     self.state['capacity']['num_adapts_activ'].append(num_adapts_activ)
    #     self.state['capacity']['num_shared_activ'].append(num_shared_activ)
    #     self.state['capacity']['ratio'].append(ratio)
    #     self.logger.print(self.state['client_id'], 'model capacity: %.3f' % (ratio))

    # def calculate_communication_costs(self, params):
    #     if self.state['num_total_params'] == 0:
    #         for dims in self.nets.shapes:
    #             params = 1
    #             for d in dims:
    #                 params *= d
    #             self.state['num_total_params'] += params

    #     num_actives = 0
    #     for i, pruned in enumerate(params):
    #         actives = torch.ne(pruned, torch.zeros_like(pruned)).float()
    #         actives = torch.sum(actives)
    #         num_actives += actives

    #     ratio = num_actives / self.state['num_total_params']
    #     self.state['communication']['num_actives'].append(num_actives)
    #     self.state['communication']['ratio'].append(ratio)
    #     self.logger.print(self.state['client_id'], 'communication cost: %.3f' % (ratio))

    def set_details(self, details):
        self.params = details

    def set_task(self, task):
        self.task = task

    def get_scores(self):
        return self.state['scores']

    def get_capacity(self):
        return self.state['capacity']

    def get_communication(self):
        return self.state['communication']

    def aggregate(self, updates):
        if self.args.sparse_comm and self.args.algorithm in ['FedWeIT']:
            client_weights = [u[0][0] for u in updates]
            client_masks = [u[0][1] for u in updates]
            client_sizes = [u[1] for u in updates]
            
            new_weights = [torch.zeros_like(torch.tensor(w)) for w in client_weights[0]]
            epsi = 1e-15
            total_sizes = [epsi for i in range(len(client_masks[0]))]

            # for mask in client_masks:
            #     print(len(mask[0]))
            for cid, mask_cid in enumerate(client_masks):
                for lid, mask_lid in enumerate(mask_cid):
                    total_sizes[lid] += sum(client_masks[cid][lid]) 
            
            for c_idx, c_weights in enumerate(client_weights): # by client
                for lidx, l_weights in enumerate(c_weights): # by layer
                    ratio = 1 / total_sizes[lidx]
                    new_weights[lidx] += l_weights * ratio
        else:
            client_weights = [u[0] for u in updates]
            client_sizes = [u[1] for u in updates]
            new_weights = [torch.zeros_like(w) for w in client_weights[0]]
            total_size = len(client_sizes)
            for c in range(len(client_weights)): # by client
                _client_weights = client_weights[c]
                for i in range(len(new_weights)): # by layer
                    new_weights[i] += _client_weights[i] * float(1 / total_size)
        return new_weights

class NetModule:
    def __init__(self, args):
        self.args = args
        self.initializer = torch.nn.init.kaiming_normal_

        self.state = {}
        self.models = []
        self.heads = []
        self.decomposed_layers = {}
        self.initial_body_weights = []
        self.initial_heads_weights = []

        self.lid = 0
        self.adaptive_factor = 3
        self.input_shape = (3, 32, 32)
        
        if self.args.base_network == 'lenet':
            self.shapes = [
                (20, 3, 5, 5),
                (50, 20, 5, 5),
                (3200, 800),
                (800, 500)]
        
        if self.args.algorithm in ['FedWeIT']:
            self.decomposed_variables = {
                'shared': [],
                'adaptive':{},
                'mask':{},
                'bias':{},
            }
            if self.args.algorithm in ['FedWeIT']:
                self.decomposed_variables['atten'] = {}
                self.decomposed_variables['from_kb'] = {}

    def init_state(self, cid):
        if self.args.algorithm in ['FedWeIT']:
            self.state = {
                'client_id':  cid,
                'decomposed_weights': {
                    'shared': [],
                    'adaptive':{},
                    'mask':{},
                    'bias':{},
                },
                'heads_weights': self.initial_heads_weights,
            }
            if self.args.algorithm in ['FedWeIT']:
                self.state['decomposed_weights']['atten'] = {}
                self.state['decomposed_weights']['from_kb'] = {}
        else:
            self.state = {
                'client_id':  cid,
                'body_weights': self.initial_body_weights,
                'heads_weights': self.initial_heads_weights,
            } 

    def save_state(self):
        self.state['heads_weights'] = []
        for h in self.heads:
            self.state['heads_weights'].append(h.state_dict())
        if self.args.algorithm in ['FedWeIT']:
            for var_type, layers in self.decomposed_variables.items():
                self.state['decomposed_weights'] = {
                    'shared': [layer.detach().cpu().numpy() for layer in self.decomposed_variables['shared']],
                    'adaptive':{tid: [layer.detach().cpu().numpy() for lid, layer in self.decomposed_variables['adaptive'][tid].items()] for tid in self.decomposed_variables['adaptive'].keys()},
                    'mask':{tid: [layer.detach().cpu().numpy() for lid, layer in self.decomposed_variables['mask'][tid].items()] for tid in self.decomposed_variables['mask'].keys()},
                    'bias':{tid: [layer.detach().cpu().numpy() for lid, layer in self.decomposed_variables['bias'][tid].items()] for tid in self.decomposed_variables['bias'].keys()},
                }
                if self.args.algorithm in ['FedWeIT']:
                    self.state['decomposed_weights']['from_kb'] = {tid: [layer.detach().cpu().numpy() for lid, layer in self.decomposed_variables['from_kb'][tid].items()] for tid in self.decomposed_variables['from_kb'].keys()}
                    self.state['decomposed_weights']['atten'] = {tid: [layer.detach().cpu().numpy() for lid, layer in self.decomposed_variables['atten'][tid].items()] for tid in self.decomposed_variables['atten'].keys()}
        else:
            self.state['body_weights'] = self.model_body.state_dict()
        
        np_save(self.args.state_dir, '{}_net.npy'.format(self.state['client_id']), self.state)

    def load_state(self, cid):
        self.state = np_load(os.path.join(self.args.state_dir, '{}_net.npy'.format(cid))).item()
        # print("len_heads_state: " + str(len(self.state['heads_weights'])))
        # print("len_heads: " + str(len(self.heads)))

        for i, h in enumerate(self.state['heads_weights']):
            self.heads[i].load_state_dict(h)

        if self.args.algorithm in ['FedWeIT']:
            for var_type, values in self.state['decomposed_weights'].items():
                if var_type == 'shared':
                    for lid, weights in enumerate(values):
                        self.decomposed_variables['shared'][lid].data = torch.tensor(weights)
                else:
                    for tid, layers in values.items():
                        for lid, weights in enumerate(layers):    
                            self.decomposed_variables[var_type][tid][lid].data = torch.tensor(weights)
        else:
            self.model_body.load_state_dict(self.state['body_weights'])

    def init_global_weights(self):
        if self.args.algorithm in ['FedWeIT']:
            global_weights = []
            for i in range(len(self.shapes)):
                global_weights.append(self.initializer(torch.empty(self.shapes[i])).numpy())
        else:
            if self.args.base_network == 'lenet':
                body = self.build_lenet_body(decomposed=False)
            global_weights = body.state_dict()
        return global_weights

    # def init_decomposed_variables(self, initial_weights):
    #     print("hello")
    #     self.decomposed_variables['shared'] = [torch.nn.Parameter(torch.tensor(initial_weights[i]), requires_grad=True) for i in range(len(self.shapes))]
    #     for tid in range(500): ## bug, sua sau
    #         print(tid)
    #         for lid in range(len(self.shapes)):
    #             var_types = ['adaptive', 'bias', 'mask'] if self.args.model == 'apd' else ['adaptive', 'bias', 'mask', 'atten', 'from_kb']
    #             for var_type in var_types:
    #                 self.create_variable(var_type, lid, tid)
    #     # print("hello1")

    def get_memory_usage_mb(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # RSS: Resident Set Size in bytes → MB

    def init_decomposed_variables(self, initial_weights):
        import time
        import psutil
        import os

        # print("hello")
        self.decomposed_variables['shared'] = [
            torch.nn.Parameter(torch.tensor(initial_weights[i]), requires_grad=True) 
            for i in range(len(self.shapes))
        ]

        mem_before = self.get_memory_usage_mb()
        for tid in range(int(self.args.num_classes/self.args.cpt)):  # bug, sửa sau
            print(f"tid={tid}")
            t0 = time.time()
            for lid in range(len(self.shapes)):
                var_types = ['adaptive', 'bias', 'mask'] if self.args.model == 'apd' else ['adaptive', 'bias', 'mask', 'atten', 'from_kb']
                for var_type in var_types:
                    self.create_variable(var_type, lid, tid)
            mem_after = self.get_memory_usage_mb()
            print(f"  RAM used: {mem_after - mem_before:.2f} MB (+{mem_after:.2f} total)")
            mem_before = mem_after

    # def create_variable(self, var_type, lid, tid=None):
    #     trainable = True
    #     if tid not in self.decomposed_variables[var_type]:
    #         self.decomposed_variables[var_type][tid] = {}

    #     if var_type == 'adaptive':
    #         # Tạo bản sao khởi tạo từ shared weight, chia theo factor
    #         init_value = self.decomposed_variables['shared'][lid].detach().clone() / self.adaptive_factor

    #     elif var_type == 'atten':
    #         shape = (int(round(self.args.num_clients * self.args.join_ratio)),)
    #         init_value = torch.zeros(shape)
    #         if tid != 0:
    #             init.kaiming_uniform_(init_value.unsqueeze(0), a=math.sqrt(5))  # cần thêm math
    #             init_value = init_value.squeeze(0)
    #         else:
    #             trainable = False

    #     elif var_type == 'from_kb':
    #         shape = list(self.shapes[lid]) + [int(round(self.args.num_clients * self.args.join_ratio))]
    #         init_value = torch.zeros(shape)
    #         if tid != 0:
    #             init.kaiming_uniform_(init_value, a=math.sqrt(5))
    #         else:
    #             trainable = False

    #     elif var_type == 'bias' and lid in [2, 3]:
    #         shape = (self.shapes[lid][-1],)
    #         init_value = torch.zeros(shape)
    #         init.kaiming_uniform_(init_value.unsqueeze(0), a=math.sqrt(5))
    #         init_value = init_value.squeeze(0)

    #     else:
    #         shape = (self.shapes[lid][0],)
    #         init_value = torch.zeros(shape)
    #         init.kaiming_uniform_(init_value.unsqueeze(0), a=math.sqrt(5))
    #         init_value = init_value.squeeze(0)

    #     var = torch.nn.Parameter(init_value, requires_grad=trainable)
    #     self.decomposed_variables[var_type][tid][lid] = var

    def create_variable(self, var_type, lid, tid=None):
        trainable = True 
        if tid not in self.decomposed_variables[var_type]:
            self.decomposed_variables[var_type][tid] = {}
        
        if var_type == 'adaptive':
            init_value = self.decomposed_variables['shared'][lid].detach().cpu().numpy()/self.adaptive_factor
        
        elif var_type == 'atten':
            shape = (int(round(self.args.num_clients*self.args.join_ratio)),)
            if tid == 0:
                trainable = False
            init_value = np.zeros(shape).astype(np.float32)
        
        elif var_type == 'from_kb':
            shape = np.concatenate([self.shapes[lid], [int(round(self.args.num_clients*self.args.join_ratio))]], axis=0)
            trainable = False
            init_value = np.zeros(shape).astype(np.float32)
        
        elif var_type == 'bias' and lid in [2,3]:
            init_value = np.zeros(self.shapes[lid][-1]).astype(np.float32)

        else:
            init_value = np.zeros(self.shapes[lid][0]).astype(np.float32)
        
        var = torch.nn.Parameter(torch.tensor(init_value), requires_grad=trainable)
        self.decomposed_variables[var_type][tid][lid] = var

    def get_variable(self, var_type, lid, tid=None):
        if var_type == 'shared':
            return self.decomposed_variables[var_type][lid]
        else:
            return self.decomposed_variables[var_type][tid][lid]

    def generate_mask(self, mask):
        with torch.no_grad():  # Không tính gradient khi cập nhật trực tiếp
            mask.copy_(torch.sigmoid(mask))
        return mask

    def get_model_by_tid(self, tid):
        if self.args.algorithm in ['FedWeIT']:
            self.switch_model_params(tid)
        return self.models[0]

    def get_trainable_variables(self, curr_task, head=True):
        if self.args.algorithm in ['FedWeIT']:
            return self.get_decomposed_trainable_variables(curr_task, retroactive=False, head=head)
        else:
            if head:
                return self.models[curr_task].parameters()
            else:
                return self.model_body.parameters()

    def get_decomposed_trainable_variables(self, curr_task, retroactive=False, head=True):
        prev_variables = ['mask', 'bias', 'adaptive'] if self.args.model == 'apd' else ['mask', 'bias', 'adaptive', 'atten']
        trainable_variables = [sw for sw in self.decomposed_variables['shared']]
        if retroactive:
            for tid in range(curr_task+1):
                for lid in range(len(self.shapes)):
                    for pvar in prev_variables:
                        if pvar == 'bias' and tid < curr_task:
                            continue
                        if pvar == 'atten' and tid == 0:
                            continue
                        trainable_variables.append(self.get_variable(pvar, lid, tid))
        else:
            for lid in range(len(self.shapes)):
                for pvar in prev_variables:
                    if pvar == 'atten' and curr_task == 0:
                        continue
                    trainable_variables.append(self.get_variable(pvar, lid, curr_task))
        if head:
            head = self.heads[0]
            trainable_variables.append(head.weight)
            trainable_variables.append(head.bias)
        return trainable_variables

    def get_body_weights(self, task_id=None):
        if self.args.algorithm in ['FedWeIT']:
            prev_weights = {}
            for lid in range(len(self.shapes)):
                prev_weights[lid] = {}
                sw = self.get_variable(var_type='shared', lid=lid).detach().cpu().numpy()
                for tid in range(task_id):
                    prev_aw = self.get_variable(var_type='adaptive', lid=lid, tid=tid).detach().cpu().numpy()
                    prev_mask = self.get_variable(var_type='mask', lid=lid, tid=tid).detach().cpu().numpy()
                    prev_mask_sig = self.generate_mask(torch.tensor(prev_mask)).detach().cpu().numpy()
                    #################################################
                    num_dims = sw.ndim

                    for _ in range(1, num_dims):  
                        prev_mask_sig = np.expand_dims(prev_mask_sig, axis=-1)
                        
                    prev_weights[lid][tid] = sw * prev_mask_sig + prev_aw
                    #################################################
            return prev_weights
        else:
            return self.model_body.state_dict()

    def set_body_weights(self, body_weights):
        if self.args.algorithm in ['FedWeIT']:
            for lid, wgt in enumerate(body_weights):
                sw = self.get_variable('shared', lid)
                sw.data = torch.tensor(wgt)
        else:
            self.model_body.load_state_dict(body_weights)
    
    def switch_model_params(self, tid):
        for lid, dlay in self.decomposed_layers.items():
            dlay.sw = self.get_variable('shared', lid)
            dlay.aw = self.get_variable('adaptive', lid, tid)
            dlay.bias = self.get_variable('bias', lid, tid)
            dlay.mask = self.generate_mask(self.get_variable('mask', lid, tid))
            if self.args.algorithm in ['FedWeIT']:
                dlay.atten = self.get_variable('atten', lid, tid) 
                dlay.aw_kb = self.get_variable('from_kb', lid, tid) 

    def add_head(self, body):
        head = nn.Linear(body[-2].out_features, self.args.num_classes)
        self.heads.append(head)
        self.initial_heads_weights.append(head.state_dict())
        return nn.Sequential(body, head) # multiheaded model

    def build_lenet(self, initial_weights, decomposed=False):
        self.models = []
        # print("hello")
        self.model_body = self.build_lenet_body(initial_weights, decomposed=decomposed)
        self.set_body_weights(initial_weights)
        # print("hello 1")
        self.initial_body_weights = initial_weights
        for i in range(1):
            # print("hello 2")
            self.models.append(self.add_head(self.model_body))

    def build_lenet_body(self, initial_weights=None, decomposed=False):
        if decomposed:
            self.init_decomposed_variables(initial_weights)
            tid = 0
            layers = []
            for lid in [0, 1]:
                self.decomposed_layers[self.lid] = self.conv_decomposed(lid, tid,
                    filters = self.shapes[lid][0],
                    kernel_size = (self.shapes[lid][-1], self.shapes[lid][-2]),
                    strides = (1,1),
                    padding = 'same',
                    args = self.args)
                layers.append(self.decomposed_layers[self.lid])
                layers.append(nn.ReLU())
                self.lid += 1
                # layers.append(nn.LocalResponseNorm(4, alpha=0.001/9.0, beta=0.75, k=1.0))
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(nn.Flatten())
            for lid in [2, 3]:
                self.decomposed_layers[self.lid] = self.dense_decomposed(
                    lid, tid,
                    units=self.shapes[lid][-1],
                    args = self.args)
                layers.append(self.decomposed_layers[self.lid])
                layers.append(nn.ReLU())
                self.lid += 1
            model = nn.Sequential(*layers)
        else:
            layers = []
            layers.append(nn.Conv2d(3, 20, kernel_size=5, padding=2))
            # layers.append(nn.LocalResponseNorm(4, alpha=0.001/9.0, beta=0.75, k=1.0))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(nn.Conv2d(20, 50, kernel_size=5, padding=2))
            # layers.append(nn.LocalResponseNorm(4, alpha=0.001/9.0, beta=0.75, k=1.0))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(nn.Flatten())
            layers.append(nn.Linear(3200, 800))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(800, 500))
            layers.append(nn.ReLU())
            model = nn.Sequential(*layers)
        return model

    def conv_decomposed(self, lid, tid, filters, kernel_size, strides, padding, args):
        return  DecomposedConv(
            args        = args,
            name        = 'layer_{}'.format(lid),
            filters     = filters,
            kernel_size = kernel_size,
            strides     = strides,
            padding     = padding,
            lambda_l1   = self.args.lambda_l1,
            lambda_mask = self.args.lambda_mask,
            shared      = self.get_variable('shared', lid),
            adaptive    = self.get_variable('adaptive', lid, tid),
            from_kb     = self.get_variable('from_kb', lid, tid),
            atten       = self.get_variable('atten', lid, tid),
            bias        = self.get_variable('bias', lid, tid), use_bias=True,
            mask        = self.generate_mask(self.get_variable('mask', lid, tid)))

    def dense_decomposed(self, lid, tid, units, args):
        return DecomposedDense(
            args        = args,
            name        = 'layer_{}'.format(lid),
            units       = units,
            lambda_l1   = self.args.lambda_l1,
            lambda_mask = self.args.lambda_mask,
            shared      = self.get_variable('shared', lid),
            adaptive    = self.get_variable('adaptive', lid, tid),
            from_kb     = self.get_variable('from_kb', lid, tid),
            atten       = self.get_variable('atten', lid, tid),
            bias        = self.get_variable('bias', lid, tid), use_bias=True,
            mask        = self.generate_mask(self.get_variable('mask', lid, tid)))
    
# Layers
class DecomposedDense(nn.Module):
    """ Custom dense layer that decomposes parameters into shared and specific parameters.
    """
    def __init__(self,
                 args, 
                 units,
                 use_bias=False,
                 lambda_l1=None,
                 lambda_mask=None,
                 shared=None,
                 adaptive=None,
                 from_kb=None,
                 atten=None,
                 mask=None,
                 bias=None,
                 **kwargs):
        super(DecomposedDense, self).__init__()
        
        self.args = args
        self.units = units
        self.use_bias = use_bias
        
        self.sw = shared
        self.aw = adaptive
        self.mask = mask
        self.bias = bias
        self.aw_kb = from_kb
        self.atten = atten
        self.lambda_l1 = lambda_l1
        self.lambda_mask = lambda_mask

    @property
    def out_features(self):
        return self.units
    
    def l1_pruning(self, weights, hyp):
        hard_threshold = (weights.abs() > hyp).float()
        return weights * hard_threshold
    
    def forward(self, inputs):
        aw = self.aw if self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask if self.training else self.l1_pruning(self.mask, self.lambda_mask)
        atten = self.atten
        aw_kbs = self.aw_kb

        a = self.sw * mask.view(mask.shape[0], 1)
        b = torch.sum(aw_kbs * atten, dim=-1)
        c = aw

        # self.my_theta = self.sw * mask.view(mask.shape[0], 1, 1, 1) + aw + torch.sum(aw_kbs * atten, dim=-1)
        self.my_theta = a
        self.my_theta += b
        self.my_theta += c

        # print(inputs.shape)
        # print(self.my_theta.shape)
        outputs = torch.matmul(inputs, self.my_theta)
        
        # print(self.bias.shape)
        # print(outputs.shape)
        if self.use_bias:
            outputs = outputs + self.bias.view(1, -1)
        
        return outputs

class DecomposedConv(nn.Module):
    """ Custom conv layer that decomposes parameters into shared and specific parameters.
    """
    def __init__(self, 
                 args,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 use_bias=False,
                 lambda_l1=None,
                 lambda_mask=None,
                 shared=None,
                 adaptive=None,
                 from_kb=None,
                 atten=None,
                 mask=None,
                 bias=None,
                 **kwargs):
        super(DecomposedConv, self).__init__()
        
        self.args = args
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        # potential bug
        
        self.sw = shared
        self.aw = adaptive
        self.mask = mask
        self.bias = bias
        self.aw_kb = from_kb
        self.atten = atten
        self.lambda_l1 = lambda_l1
        self.lambda_mask = lambda_mask
    
    @property
    def out_features(self):
        return self.units
    
    def l1_pruning(self, weights, hyp):
        hard_threshold = (weights.abs() > hyp).float()
        return weights * hard_threshold
    
    def forward(self, inputs):
        aw = self.aw if self.training else self.l1_pruning(self.aw, self.lambda_l1)
        mask = self.mask if self.training else self.l1_pruning(self.mask, self.lambda_mask)
        atten = self.atten
        aw_kbs = self.aw_kb      

        # print(f"mask shape: {mask.shape}")
        # print(f"aw_kbs shape: {aw_kbs.shape}, atten shape: {atten.shape}")
        # print(f"sw shape: {self.sw.shape}")

        a = self.sw * mask.view(mask.shape[0], 1, 1, 1)
        b = torch.sum(aw_kbs * atten, dim=-1)
        c = aw

        # self.my_theta = self.sw * mask.view(mask.shape[0], 1, 1, 1) + aw + torch.sum(aw_kbs * atten, dim=-1)
        self.my_theta = a
        self.my_theta += b
        self.my_theta += c

        # print(inputs.shape)
        # print(self.my_theta.shape)
        outputs = F.conv2d(inputs, self.my_theta, stride=self.strides, padding=self.padding, dilation=self.dilation_rate)

        # print(outputs.shape)
        # print(self.bias.view(1, -1, 1, 1).shape)

        if self.use_bias:
            outputs = outputs + self.bias.view(1, -1, 1, 1)
        
        return outputs

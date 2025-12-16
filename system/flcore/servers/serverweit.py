import time
import torch
from flcore.clients.clientweit import clientWeIT
from flcore.servers.serverbase import Server
from flcore.trainmodel.fedewit_models import *
from flcore.utils_core.fedweit_utils import *
from threading import Thread
from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
import shutil

class FedWeIT(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        args.log_dir = os.path.join(args.output_path, 'logs/{}-{}'.format(args.model, args.dataset))
        args.state_dir = os.path.join(args.output_path, 'states/{}-{}'.format(args.model, args.dataset))

        # self.load_model()
        self.Budget = []
        self.client_adapts = []

        # self.logger = Logger(self.args)
        self.logger = Logger(self.args)
        self.nets = NetModule(self.args)
        self.trainh = TrainModule(self.args, self.logger, self.nets)

        self.nets.init_state(None)
        self.trainh.init_state(None)
        self.global_weights = self.nets.init_global_weights()

        self.set_clients(clientWeIT)
        print(f"Number of clients: {len(self.clients)}")
        print("heheeeeee")

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        if os.path.exists("./output_fedweit"):
            shutil.rmtree("./output_fedweit")

        for task in range(self.args.num_tasks):
            # print("Change task")
            print(f"\n================ Current Task: {task} =================")
            if task == 0:
                 # update labels info. for the first task
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                    
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            else:
                self.current_task = task
                
                torch.cuda.empty_cache()
                for i in range(len(self.clients)):
                    
                    if self.args.dataset == 'IMAGENET1k':
                        train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    elif self.args.dataset == 'CIFAR100':
                        train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                    else:
                        raise NotImplementedError("Not supported dataset")

                    # update dataset
                    # assert (self.users[i].id == id)
                    self.clients[i].next_task(train_data, label_info) # assign dataloader for new data
                    
                # update labels info.
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))

                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            for i in range(self.global_rounds):
                
                print("Change glob round")
                glob_iter = i + self.global_rounds * task
                self.updates = []
                self.curr_round = glob_iter+1
                self.is_last_round = i==0
                if self.is_last_round:
                    self.client_adapts = []
                s_t = time.time()
                self.selected_clients = self.select_clients()
                # self.send_models()

                if i%self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    self.eval(task=task, glob_iter=glob_iter, flag="global")

                for client in self.selected_clients:
                    update = client.train_one_round(client.id, glob_iter, True, self.get_weights(), self.get_adapts(glob_iter=glob_iter))
                    if not update == None:
                        self.updates.append(update)
                        if self.is_last_round:
                            self.client_adapts.append(client.get_adaptives())

                aggr = self.trainh.aggregate(self.updates)
                self.set_weights(aggr)

                self.Budget.append(time.time() - s_t)
                print('-'*25, 'time cost', '-'*25, self.Budget[-1])

    def get_weights(self):
        return self.global_weights

    def set_weights(self, weights):
        self.global_weights = weights

    def get_adapts(self, glob_iter):
        if glob_iter%self.global_rounds==1 and not glob_iter==1:
            from_kb = []
            for lid, shape in enumerate(self.nets.shapes):
                shape = np.concatenate([self.nets.shapes[lid],[int(round(self.args.num_clients*self.join_ratio))]], axis=0)
                from_kb_l = np.zeros(shape)
                for cid, ca in enumerate(self.client_adapts):
                    try:
                        if len(shape)==5:
                            from_kb_l[:,:,:,:,cid] = ca[lid]
                        else:
                            from_kb_l[:,:,cid] = ca[lid]
                    except:
                        pdb.set_trace()           
                from_kb.append(from_kb_l)
            return from_kb
        else:
            return None
        
    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            
            if self.args.dataset == 'IMAGENET1k':
                train_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
            elif self.args.dataset == 'CIFAR100':
                train_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
            else:
                raise NotImplementedError("Not supported dataset")
            
            # count total samples (accumulative)
            id = i
            client = clientObj(self.args, 
                            id=i,
                            train_data=train_data, 
                            initial_weights=self.global_weights)
            self.clients.append(client)
            
            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])

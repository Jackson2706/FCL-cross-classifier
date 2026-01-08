from flcore.clients.clientfedprox import clientProx
from flcore.servers.serveravg import FedAvg


class FedProx(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientProx)
import copy

import torch
from flcore.clients.clientfednova import clientNova
from flcore.servers.serveravg import FedAvg


class FedNova(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientNova)

    def aggregate_parameters(self):
        """
        FedNova Aggregation logic
        """
        assert (len(self.selected_clients) > 0)

        # 1. Collect Client Metadata (taus and sample counts)
        # --------------------------------------------------
        # n_k: number of samples on client k
        nk_list = [len(c.train_data) for c in self.selected_clients]
        total_n = sum(nk_list)
        
        # p_k: relative weight of client k
        pk_list = [n / total_n for n in nk_list]
        
        # tau_k: number of local steps performed by client k
        tau_list = [c.local_epochs for c in self.selected_clients]

        # 2. Calculate Effective Tau (tau_eff)
        # ------------------------------------
        # tau_eff = sum(p_k * tau_k)
        tau_eff = sum([p * t for p, t in zip(pk_list, tau_list)])

        # 3. Aggregate Normalized Updates
        # -------------------------------
        # Formula: w_new = w_old - sum [ p_k * (tau_eff / tau_k) * (w_old - w_client) ]
        
        # Create a zeroed structure for the accumulated update
        global_model_params = list(self.global_model.parameters())
        accumulated_update = [torch.zeros_like(p) for p in global_model_params]

        for idx, client in enumerate(self.selected_clients):
            # Calculate the scaling factor for this client
            # scale = p_k * (tau_eff / tau_k)
            # Note: We handle division by zero just in case, though unlikely in training
            if tau_list[idx] == 0:
                scale = 0
            else:
                scale = pk_list[idx] * (tau_eff / tau_list[idx])

            # Calculate client update: (w_old - w_client)
            for i, (g_param, c_param) in enumerate(zip(global_model_params, client.model.parameters())):
                update = g_param.data - c_param.data.to(self.device)
                
                # Add weighted normalized update to accumulator
                accumulated_update[i] += update * scale

        # 4. Apply Update to Global Model
        # -------------------------------
        for i, param in enumerate(global_model_params):
            # w_new = w_old - accumulated_update
            param.data = param.data - accumulated_update[i]

    # Note: In flcore, 'receive_models' often calls 'aggregate_parameters'.
    # You generally don't need to override receive_models unless it does something unique.
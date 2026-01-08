# system/flcore/servers/serverLANDER.py
import copy

import torch
from flcore.clients.clientLANDER import clientLANDER
from flcore.servers.serverbase import Server

# from flcore.utils.text_encoder import get_text_anchors

class LANDERServer(Server):
    """
    LANDER server:
    - Precompute/broadcast LTE anchors from class names (once).
    - Maintain previous global as teacher for optional KD synergy.
    """
    def __init__(self, args, times):
        super().__init__(args, times)
        try:
            self.set_slow_clients()
        except Exception:
            pass
        self.set_clients(clientLANDER)

        self.best_global = None
        self._anchors = None

    def _ensure_anchors(self):
        if self._anchors is not None:
            return self._anchors
        # Get class names from any client's dataset if available
        class_names = None
        if self.clients and hasattr(self.clients[0].trainloader.dataset, "classes"):
            class_names = list(self.clients[0].trainloader.dataset.classes)
        else:
            C = getattr(self.args, "num_classes", 100)
            class_names = [f"class_{i}" for i in range(C)]
        anchors = get_text_anchors(class_names, model_name=getattr(self.args, "lander_text_encoder", "clip-ViT-B-32"),
                                   device=self.device,
                                   template=getattr(self.args, "lander_text_template", "a photo of a {}"))
        self._anchors = anchors.cpu()
        return self._anchors

    def train(self):
        anchors = self._ensure_anchors()

        for round in range(self.global_rounds):
            self.selected_clients = self.select_clients()

            # Broadcast global, teacher, and anchors
            teacher = self.best_global
            for c in self.selected_clients:
                try:
                    c.set_teacher(teacher)
                    c.set_anchors(anchors.to(self.device))
                except Exception:
                    pass
                self.send_models(c)

            self.receive_models()
            self.aggregate_parameters()

            if hasattr(self, "global_test"):
                acc = self.global_test()
            else:
                acc = None
            if acc is None or (self.rs_test_acc and self.rs_test_acc[-1] >= max(self.rs_test_acc)):
                self.best_global = copy.deepcopy(self.global_model)

            self.print_result(round)

        self.save_results()
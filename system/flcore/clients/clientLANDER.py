# system/flcore/clients/clientLANDER.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from flcore.criterion.distillation import \
    kl_divergence_with_temperature as kd_kl
from flcore.criterion.lander_losses import bounding_loss
from torch.utils.data import ConcatDataset, DataLoader

# from flcore.utils.text_encoder import get_text_anchors

class clientLANDER(Client):
    """
    LANDER: Label-Text centered, data-free knowledge transfer via LTE anchors.
    - During training: constrain features around class text anchors (Bounding Loss)
    - Optional KD from server teacher
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.num_classes = getattr(args, "num_classes", 100)
        self.T = getattr(args, "lander_T", 2.0)
        self.lambda_bound = getattr(args, "lander_lambda_bound", 1.0)
        self.radius = getattr(args, "lander_radius", 0.5)
        self.kd_coef = getattr(args, "lander_kd", 0.5)
        self.text_model = getattr(args, "lander_text_encoder", "clip-ViT-B-32")
        self.text_template = getattr(args, "lander_text_template", "a photo of a {}")
        self._teacher_model = None

        # cached anchors (C, D)
        self._anchors = None
        self._feature_dim = getattr(args, "feature_dim", 512)

    def set_teacher(self, model_or_none):
        self._teacher_model = model_or_none

    def set_anchors(self, anchors):
        """Server can broadcast precomputed anchors (C, D)."""
        self._anchors = anchors

    def _ensure_anchors(self):
        if self._anchors is not None:
            return self._anchors
        # try to get class names from dataset if available
        class_names = None
        if hasattr(self.trainloader.dataset, "classes"):
            class_names = list(self.trainloader.dataset.classes)
        else:
            class_names = [f"class_{i}" for i in range(self.num_classes)]
        anchors = get_text_anchors(class_names, model_name=self.text_model,
                                   device=self.device, template=self.text_template)
        if anchors.shape[1] != self._feature_dim:
            # project to feature dim with a linear layer (no-grad) if mismatch
            W = torch.randn(anchors.shape[1], self._feature_dim, device=anchors.device)
            anchors = anchors @ F.normalize(W, dim=0)
        self._anchors = anchors
        return anchors

    def _forward_features(self, x):
        if hasattr(self.model, "forward_features"):
            return self.model.forward_features(x)
        # fallback: use logits as features
        return self.model(x)

    def train(self):
        self.model.to(self.device).train()
        anchors = self._ensure_anchors().to(self.device)

        # if you maintain exemplar memory in your branch, you can add it here
        dataset = self.trainloader.dataset
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        opt = self.optimizer
        for _ in range(self.local_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                feat = self._forward_features(xb)
                if feat.dim() > 2:
                    feat = torch.flatten(feat, 1)

                logits = self.model(xb) if hasattr(self.model, "classifier") else feat
                ce = F.cross_entropy(logits, yb)

                b_loss = bounding_loss(feat, anchors, yb, radius=self.radius)

                loss = ce + self.lambda_bound * b_loss

                if self._teacher_model is not None:
                    with torch.no_grad():
                        t_logits = self._teacher_model(xb)
                    loss = loss + self.kd_coef * kd_kl(logits, t_logits, T=self.T)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()

        return
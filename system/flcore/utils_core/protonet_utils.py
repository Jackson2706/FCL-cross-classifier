import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class ProtoNet_Loss(nn.Module):
    def __init__(self, n_support):
        super(ProtoNet_Loss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)

(''
 ''
 ''
 '')
class OnPro_Loss(nn.Module):
    def __init__(self):
        self.loss = nn.MSELoss()

    def forward(self, args, img, rec):
        rec_loss = self.rec_loss(img, rec)
        total_loss = rec_loss

        return total_loss


"""
    Supplementary Functions:
"""
def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_query):
    """
    Compute prototypical loss and accuracy using fixed n_query per class.

    Args:
        input (Tensor): Model output of shape [N, D]
        target (Tensor): Ground truth labels of shape [N]
        n_query (int): Number of query samples per class

    Returns:
        Tuple[Tensor, Tensor]: loss and accuracy
    """
    classes = torch.unique(target)
    n_classes = len(classes)

    support_idxs = []
    query_idxs = []

    for cls in classes:
        cls_idxs = (target == cls).nonzero(as_tuple=True)[0]
        if len(cls_idxs) < n_query + 1:
            raise ValueError(f"Not enough samples for class {cls.item()}: need > {n_query}, got {len(cls_idxs)}")
        query_idxs.append(cls_idxs[-n_query:])
        support_idxs.append(cls_idxs[:-n_query])

    support_idxs = torch.cat(support_idxs)
    query_idxs = torch.cat(query_idxs)

    support = input[support_idxs]
    query = input[query_idxs]

    prototypes = []
    for cls in classes:
        cls_support = support[(target[support_idxs] == cls)]
        prototypes.append(cls_support.mean(0))
    prototypes = torch.stack(prototypes)

    dists = euclidean_dist(query, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)

    target_inds = torch.arange(n_classes, device=target.device).repeat_interleave(n_query)
    loss = -log_p_y[range(len(query_idxs)), target_inds].mean()

    pred = log_p_y.argmax(dim=1)
    acc = (pred == target_inds).float().mean()

    return loss, acc
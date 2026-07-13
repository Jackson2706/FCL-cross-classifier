"""Signal providers shared by legacy BN loss and reliability scoring."""

import torch
from torch import nn


class BNFeatureHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_feature = None

    def hook_fn(self, module, inputs, output):
        del module, output
        self.r_feature = inputs[0]

    def remove(self):
        self.hook.remove()


def bn_feature_distance(teacher_model, synthetic_images, per_sample=False):
    """Compare synthetic BN features with a teacher's running statistics.

    The batch reduction exactly preserves the historical ``get_bn_loss``
    expression.  The per-sample form computes spatial moments independently and
    is used only by active reliability modes.
    """

    layers = [m for m in teacher_model.modules() if isinstance(m, nn.BatchNorm2d)]
    hooks = [BNFeatureHook(layer) for layer in layers]
    try:
        teacher_model(synthetic_images)
        if per_sample:
            distance = synthetic_images.new_zeros(synthetic_images.shape[0])
        else:
            distance = synthetic_images.new_tensor(0.0)
        for hook, layer in zip(hooks, layers):
            feature = hook.r_feature
            if per_sample:
                generated_mean = feature.mean(dim=(2, 3))
                generated_var = feature.var(dim=(2, 3), unbiased=False)
                distance = distance + torch.linalg.vector_norm(
                    generated_mean - layer.running_mean.unsqueeze(0), dim=1
                ) + torch.linalg.vector_norm(
                    generated_var - layer.running_var.unsqueeze(0), dim=1
                )
            else:
                generated_mean = feature.mean(dim=(0, 2, 3))
                generated_var = feature.var(dim=(0, 2, 3), unbiased=False)
                distance = distance + torch.norm(
                    generated_mean - layer.running_mean, 2
                ) + torch.norm(generated_var - layer.running_var, 2)
        if per_sample and layers:
            # The legacy batch loss is a layer sum.  Reliability needs a scale
            # comparable across architectures, so its per-sample distance is a
            # mean per BN feature channel.
            distance = distance / sum(layer.num_features for layer in layers)
        return distance
    finally:
        for hook in hooks:
            hook.remove()

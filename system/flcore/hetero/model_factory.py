"""Factory for CIFAR classifiers with a common ``base``/``head`` interface.

Phase 6a intentionally supports FedAvg only when every client and the server use
the same architecture.  Cross-architecture aggregation belongs to Phase 6b.
"""

from types import SimpleNamespace

import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock, ResNet

from flcore.trainmodel.models import BaseHeadSplit, FedAvgCNN


HETERO_CONFIG_DEFAULTS = {
    "model_heterogeneity": False,
    "client_model_pool": ["resnet18"],
    "server_model": "resnet18",
    "aggregation_mode": "fedavg",
}

SUPPORTED_CLIENT_MODELS = {
    "small_cnn",
    "resnet18",
    "mobilenetv2",
    "lightweight_resnet",
}


def _canonical_name(name):
    canonical = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "smallcnn": "small_cnn",
        "cnn": "small_cnn",
        "resnet18": "resnet18",
        "mobilenetv2": "mobilenetv2",
        "lightweightresnet": "lightweight_resnet",
        "resnet10": "lightweight_resnet",
    }
    try:
        return aliases[canonical]
    except KeyError as exc:
        raise ValueError(
            f"Unknown client model {name!r}. Choose from: "
            f"{sorted(SUPPORTED_CLIENT_MODELS)}"
        ) from exc


def _split_fc_model(model):
    head = model.fc
    model.fc = nn.Identity()
    return BaseHeadSplit(model, head)


def _build_resnet18(num_classes):
    # This deliberately mirrors main.py's legacy ResNet18 construction.  The
    # legacy path itself never calls this factory when heterogeneity is disabled.
    return _split_fc_model(
        torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    )


def _build_small_cnn(num_classes):
    # For 32x32 CIFAR inputs the two 5x5 conv/pool stages flatten to 64*5*5.
    model = FedAvgCNN(in_features=3, num_classes=num_classes, dim=1600)
    return _split_fc_model(model)


def _build_mobilenetv2(num_classes):
    model = torchvision.models.mobilenet_v2(weights=None, num_classes=num_classes)
    # CIFAR adaptation: retain spatial detail by removing the ImageNet stem's
    # stride-2 downsampling.  MobileNet's adaptive pooling still yields a
    # 1280-dimensional feature vector.  The full dropout+linear classifier is
    # the head; ``base`` ends after pooling/flattening.
    model.features[0][0].stride = (1, 1)
    head = model.classifier
    model.classifier = nn.Identity()
    return BaseHeadSplit(model, head)


def _build_lightweight_resnet(num_classes):
    # A CIFAR-adapted ResNet-10: one BasicBlock in each of four stages instead
    # of ResNet-18's two, with a 3x3 stride-1 stem and no max-pool.
    model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=num_classes)
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    return _split_fc_model(model)


def build_client_model(name, args):
    """Build a BaseHeadSplit-wrapped classifier for ``args.num_classes``."""

    name = _canonical_name(name)
    num_classes = int(args.num_classes)
    builders = {
        "small_cnn": _build_small_cnn,
        "resnet18": _build_resnet18,
        "mobilenetv2": _build_mobilenetv2,
        "lightweight_resnet": _build_lightweight_resnet,
    }
    model = builders[name](num_classes)
    model.model_type = name
    return model.to(getattr(args, "device", "cpu"))


def resolve_heterogeneity_config(args):
    """Resolve and validate Phase-6a model-assignment configuration."""

    enabled = bool(getattr(args, "model_heterogeneity", False))
    if not enabled:
        # All other heterogeneity fields are inert in the compatibility path.
        return SimpleNamespace(
            enabled=False,
            client_model_pool=list(HETERO_CONFIG_DEFAULTS["client_model_pool"]),
            server_model=HETERO_CONFIG_DEFAULTS["server_model"],
            aggregation_mode=HETERO_CONFIG_DEFAULTS["aggregation_mode"],
            distinct_models=list(HETERO_CONFIG_DEFAULTS["client_model_pool"]),
        )

    raw_pool = getattr(args, "client_model_pool", ["resnet18"])
    if isinstance(raw_pool, str):
        raw_pool = [raw_pool]
    if not raw_pool:
        raise ValueError("client_model_pool must contain at least one architecture")

    pool = [_canonical_name(name) for name in raw_pool]
    server_model = _canonical_name(getattr(args, "server_model", "resnet18"))
    aggregation_mode = str(getattr(args, "aggregation_mode", "fedavg")).lower()
    distinct = sorted(set(pool))

    if enabled and len(distinct) > 1:
        if aggregation_mode == "fedavg":
            raise ValueError(
                "A heterogeneous client_model_pool cannot use aggregation_mode=fedavg; "
                "it needs aggregation_mode=logit_distillation or synthetic_distillation "
                "(Phase 6b)."
            )
        raise NotImplementedError(
            f"aggregation_mode={aggregation_mode} for a heterogeneous client_model_pool "
            "is reserved for Phase 6b distillation aggregation."
        )

    if enabled and aggregation_mode != "fedavg":
        raise NotImplementedError(
            f"aggregation_mode={aggregation_mode} is reserved for Phase 6b; Phase 6a "
            "runs homogeneous pools with aggregation_mode=fedavg."
        )
    if enabled and server_model != distinct[0]:
        raise ValueError(
            "FedAvg requires server_model to match the homogeneous client_model_pool; "
            f"got server_model={server_model!r} and client model {distinct[0]!r}."
        )

    return SimpleNamespace(
        enabled=enabled,
        client_model_pool=pool,
        server_model=server_model,
        aggregation_mode=aggregation_mode,
        distinct_models=distinct,
    )

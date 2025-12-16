python3 system/main.py --cfp ./hparams/cifar100/FedAvg_cifar100.json --seval --teval --wandb True
python3 system/main.py --cfp ./hparams/cifar100/FedDBE_cifar100.json --seval --teval --wandb True
python3 system/main.py --cfp ./hparams/cifar100/FedALA_cifar100.json --seval --teval --wandb True
python3 system/main.py --cfp ./hparams/cifar100/FedSTGM_cifar100.json --seval --teval --wandb True
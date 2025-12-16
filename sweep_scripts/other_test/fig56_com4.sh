python system/main.py --cfp ./hparams/cifar100/FedALA_cifar100.json --seval --wandb True
python system/main.py --cfp ./hparams/cifar100/FedALA_cifar100.json --seval --wandb True --cpt 20 --note 20classes
python system/main.py --cfp ./hparams/imagenet1k/FedALA.json --seval --wandb True
python system/main.py --cfp ./hparams/imagenet1k/FedDBE.json --seval --wandb True --cpt 20 --note 20classes
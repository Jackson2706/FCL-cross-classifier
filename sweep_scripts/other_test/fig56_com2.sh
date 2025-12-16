python3 system/main.py --cfp ./hparams/imagenet1k/FedAvg.json --seval --wandb True --cpt 20 --note 20classes
python3 system/main.py --cfp ./hparams/imagenet1k/FedDBE.json --seval --wandb True --cpt 20 --note 20classes
python3 system/main.py --cfp ./hparams/imagenet1k/FedSTGM.json --seval --wandb True --cpt 20 --note 20classes
python3 system/main.py --cfp ./hparams/cifar100/FedDBE_cifar100.json --seval --wandb True --cpt 20 --note 20classes
python3 system/main.py --cfp ./hparams/cifar100/FedAvg_cifar100.json --seval --wandb True --cpt 20 --note 20classes
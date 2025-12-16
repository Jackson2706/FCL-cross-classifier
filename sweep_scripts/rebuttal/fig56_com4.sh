python3 system/main.py --cfp ./hparams/cifar100/FedAvg_cifar100.json --seval --teval --wandb True --cpt 20 --nt 50 --note 20classes
python3 system/main.py --cfp ./hparams/cifar100/FedDBE_cifar100.json --seval --teval --wandb True --cpt 20 --nt 50 --note 20classes
python3 system/main.py --cfp ./hparams/cifar100/FedALA_cifar100.json --seval --teval --wandb True --cpt 20 --nt 50 --note 20classes
python3 system/main.py --cfp ./hparams/cifar100/FedSTGM_cifar100.json --seval --teval --wandb True --cpt 20 --nt 50 --note 20classes
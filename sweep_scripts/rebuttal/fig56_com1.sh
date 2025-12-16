python3 system/main.py --cfp ./hparams/imagenet1k/FedAvg.json --seval --teval --wandb True
python3 system/main.py --cfp ./hparams/imagenet1k/FedDBE.json --seval --teval --wandb True
python3 system/main.py --cfp ./hparams/imagenet1k/FedALA.json --seval --teval --wandb True
python3 system/main.py --cfp ./hparams/imagenet1k/FedSTGM.json --seval --teval --wandb True
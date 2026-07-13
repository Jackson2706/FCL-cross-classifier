# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 10
# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 50
# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100
# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 200
# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 500

# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type mlp
# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type light_cnn
# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type advanced
# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type resnet

# python3 system/main.py --cfp ./hparams/wilds/AFFCL_wilds.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type resnet
# python3 system/main.py --cfp ./hparams/wilds/FedAvg_wilds.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type resnet
# python3 system/main.py --cfp ./hparams/wilds/FedTARGET_wilds.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type resnet
# python3 system/main.py --cfp ./hparams/wilds/Ours_v2_wilds.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type resnet

# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type advanced --num_clients 20
# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type advanced --num_clients 50
# python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --generated_samples_per_class 100 --gen_type advanced --num_clients 100
python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --simulate_bad_clients True --use_filter False --run_name "Robustness_20Pct_NoFilter"
python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --simulate_bad_clients True --use_filter False --run_name "Robustness_40Pct_NoFilter"
python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --simulate_bad_clients True --use_filter True --filter_threshold 1.5 --run_name "Robustness_20Pct_WithFilter"
python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --simulate_bad_clients True --use_filter True --filter_threshold 1.5 --run_name "Robustness_40Pct_WithFilter"
python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --simulate_bad_clients False --replay_ratio 0.25 --run_name "RatioAblation_0.25_ExtremeSyn"
python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --simulate_bad_clients False --replay_ratio 0.5 --run_name "RatioAblation_0.5_HighSyn"
python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --simulate_bad_clients False --replay_ratio 2.0 --run_name "RatioAblation_2.0_LowSyn"
python3 system/main.py --cfp ./hparams/cifar100/Ours_v2_cifar100.json --wandb True --offlog True --log True --note final --simulate_bad_clients False --replay_ratio 4.0 --run_name "RatioAblation_4.0_ExtremeLowSyn"
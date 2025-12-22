# Federated Continual Learning Benchmark

## Overview
Federated Continual Learning (FCL) Benchmark is a standardized evaluation framework for assessing continual learning methods in federated settings. It provides datasets, evaluation metrics, and baseline implementations to facilitate research in FCL.

## Features
- **Diverse Datasets**: Supports multiple datasets commonly used in CFL research.
- **Baseline Models**: Includes various baseline models for comparison.
- **Customizable**: Easily extendable for new datasets and algorithms.
- **Federated Learning Simulation**: Implements a federated learning environment for continual learning.
- **Metrics & Logging**: Provides standardized metrics for evaluating performance over time.

## Installation
```sh
# Clone the repository
git clone https://github.com/chickbong221/FCL.git
cd FCL

# Create a virtual environment (optional but recommended)
python -m venv .env
source .env/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download and preprocess data
python dataset/cifar100_npy.py
python dataset/class_order_gen.py
```

Explains how to download & process ImageNet-1K train/val dataset for using as a dataset
Download ImageNet-1K train/val dataset
python unpack.py # make clean file trees of ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar
python preprocess.py # Preprocess and save train and val data of each class as 1 .npy file, ready for use
Delete the zip file and file trees from unpack.py to free space
gdown 1i0ok3LT5_mYmFWaN7wlkpHsitUngGJ8z

## Usage
### Running an Experiment
```sh
# ImageNet1k
python3 system/main.py --cfp ./hparams/imagenet1k/FedSTGM.json 
python3 system/main.py --cfp ./hparams/imagenet1k/AFFCL.json
python3 system/main.py --cfp ./hparams/imagenet1k/FedWeIT.json
python3 system/main.py --cfp ./hparams/imagenet1k/FedAS.json
python3 system/main.py --cfp ./hparams/imagenet1k/FedALA.json
python3 system/main.py --cfp ./hparams/imagenet1k/FedDBE.json
python3 system/main.py --cfp ./hparams/imagenet1k/FedAvg.json
python3 system/main.py --cfp ./hparams/imagenet1k/FedTARGET.json
python3 system/main.py --cfp ./hparams/imagenet1k/FedL2P.json
python3 system/main.py --cfp ./hparams/imagenet1k/FedAvg.json --cpt 20 --nt 50 --log True --note 20classes --wandb True

python3 system/main.py --cfp ./hparams/imagenet1k/FedALA.json  --log True --note st --wandb True --teval 
python3 system/main.py --cfp ./hparams/imagenet1k/FedTARGET.json --cpt 20 --nt 50 --log True --note 20classes_st --wandb True --teval

# Cifar100
python3 system/main.py --cfp ./hparams/cifar100/FedAvg_cifar100.json --wandb True --offlog True --log True --note final
python3 system/main.py --cfp ./hparams/cifar100/FedAS_cifar100.json --wandb True --offlog True --log True --note final
python3 system/main.py --cfp ./hparams/cifar100/FedALA_cifar100.json --wandb True --offlog True --log True --note final
python3 system/main.py --cfp ./hparams/cifar100/FedDBE_cifar100.json --wandb True --offlog True --log True --note final
python3 system/main.py --cfp ./hparams/cifar100/FedTARGET_cifar100.json --wandb True --offlog True --log True --note final
python3 system/main.py --cfp ./hparams/cifar100/AFFCL_cifar100.json --wandb True --offlog True --log True --note final
python3 system/main.py --cfp ./hparams/cifar100/FedAS_cifar100.json --cpt 20 --nt 15 --log True --offlog True --wandb True --note 20classes

python3 system/main.py --cfp ./hparams/cifar100/FedAvg_cifar100.json --cpt 20 --nt 15 --log True  --wandb True --note 20classes_st --teval 

# Cifar10
python3 system/main.py --cfp ./hparams/cifar10/FedAvg_cifar10.json
python3 system/main.py --cfp ./hparams/cifar10/FedSTGM_cifar10.json
```

Test gradient angle
--log True --offlog True --wandb True --note 20clients

Sweep
```sh
bash scripts/sweep_STGM_scripts/computer1_part3.sh
bash scripts/sweep_STGM_scripts/computer3_gpu0_job0.sh
bash scripts/sweep_STGM_scripts/computer3_gpu1_job0.sh
```

## Benchmarked Algorithms
- **FedAvg** (Federated Averaging)
- **AF-FCL** 
- **FedWeIT** 
- **FedALA** 
- **FedAS**
- **FedDBE**

## Datasets
- PMNIST
- CIFAR-100
- IMAGENET1k

## Contributing
We welcome contributions! Please check our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or collaborations, please open an issue or reach out to `Anh-Duong-dep-trai@example.com`.

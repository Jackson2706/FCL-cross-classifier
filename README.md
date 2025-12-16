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
git fetch origin
git checkout -b PFLlib-based origin/PFLlib-based

cd FCL

# Create a virtual environment (optional but recommended)
python -m venv .env
source .env/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download and preprocess data (CIFAR100 and CIFAR10)
cd dataset
python cifar100_npy.py

# Download ImageNet1k
gdown 1i0ok3LT5_mYmFWaN7wlkpHsitUngGJ8z
```
With ImageNet1K, unzip the downloaded file. Put .npy files in \FCL\dataset\imagenet1k-classes (carefully check the name please.)

## Usage
### Running an Experiment
```sh
# ImageNet1k
cd FCL
python3 system/main.py --cfp ./hparams/imagenet1k/FedSTGM.json
python3 system/main.py --cfp ./hparams/imagenet1k/FedAvg.json 

# Setting for 20 classes per task 
python3 system/main.py --cfp ./hparams/imagenet1k/FedSTGM.json --cpt 20 --nt 50 --log True --note 20classes --wandb True --teval

# Cifar100
python3 system/main.py --cfp ./hparams/cifar100/FedSTGM_cifar100.json --wandb True --offlog True --log True --note final

# Setting for 20 classes per task 
python3 system/main.py --cfp ./hparams/cifar100/FedSTGM_cifar100.json --cpt 20 --nt 15 --log True  --wandb True --note 20classes_st --teval 

# Cifar10
python3 system/main.py --cfp ./hparams/cifar10/FedAvg_cifar10.json
python3 system/main.py --cfp ./hparams/cifar10/FedSTGM_cifar10.json
```
Sweep
```sh
bash scripts/sweep_STGM_scripts/computer1_part3.sh
bash scripts/sweep_STGM_scripts/computer3_gpu0_job0.sh
bash scripts/sweep_STGM_scripts/computer3_gpu1_job0.sh
```

## Benchmarked Algorithms
- **FedSTGM**
- **AF-FCL** 
- **FedWeIT**
- **FedL2P**
- **FCIL** 
- **FedTARGET**
- **FedALA** 
- **FedAS**
- **FedDBE**
- **FedAvg**

## Datasets
- CIFAR-10
- CIFAR-100
- IMAGENET1k

## Metrics
- Average Accuracy
- Forgetting Average# FCL

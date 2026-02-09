# Federated Learning Experiments

This repository contains the implementation for running federated learning experiments as described in the paper.

---

## Dataset Preparation

The dataset generation process is described in detail in the paper. To prepare the dataset for experiments, follow the steps below:

1. Download the raw dataset file as specified in the paper.
2. Partition the dataset among multiple clients according to the data distribution strategy described in the paper.
3. This process generates a JSON file containing client-wise data.
4. Convert the generated JSON file into **LEAF format**, which is required for running the experiments.

---

## Installation

Before running the code, install the required dependencies:

pip install -r requirements.txt


## Running the Code

Use the following command to run the training script:

python3 -u main.py --dataset=$enter dataset name$ --optimizer='fedavg' \
    --learning_rate=0.01 --num_rounds=1000 --clients_per_round=10 \
    --eval_every=1 --batch_size=$enter batch size$ \
    --num_epochs=$enter epoch$ \
    --model=$'enter model name'$ \
    --drop_percent=$2 \
    --clientsel_algo='submodular'


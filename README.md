**Dataset Preparation**

The dataset generation process is described in detail in the paper. To prepare the data for experiments, follow these steps:
1) Download the raw dataset file as specified in the paper.
2) Partition the dataset among multiple clients according to the data distribution strategy described in the paper.
3) This process will generate a JSON file containing the client-wise data.
4) Convert the generated JSON file into LEAF format, which is required for running the experiments.

To run the code, use the following command:

python3 -u main.py --dataset=$name$ --optimizer='fedavg' \
    --learning_rate=0.01 --num_rounds=1000 --clients_per_round=10 \
    --eval_every=1 --batch_size=$add batch size$ \
    --num_epochs=$enter epoch$ \
    --model=$'add model name'$ \
    --drop_percent=$2 \
    --clientsel_algo='submodular'

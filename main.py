import numpy as np
import argparse
import importlib
import random
import os
import time
import tensorflow as tf
from flearn.utils.model_utils import read_data

# =========================================================
# GLOBAL PARAMETERS
# =========================================================
OPTIMIZERS = [
    'fedavg',
    'fedprox',
    'feddane',
    'fedddane',
    'fedsgd',
    'fedprox_origin'
]

DATASETS = [
    'celeba',
    'sent140',
    'nist',
    'shakespeare',
    'mnist',
    'mnist_custom',         
    'cifar10_custom',
    'synthetic_iid',
    'synthetic_0_0',
    'synthetic_0.5_0.5',
    'synthetic_1_1',
    'synthetic_cluster'
]

MODEL_PARAMS = {
    'sent140.bag_dnn': (2,),
    'sent140.stacked_lstm': (25, 2, 100),
    'sent140.stacked_lstm_no_embeddings': (25, 2, 100),

    'nist.mclr': (26,),
    'nist.cnn': (10,),

    'mnist.mclr': (10,),

    # mnist_custom uses existing models
    'mnist_custom.mclr': (10,),
    'mnist_custom.cnn': (10,),

    'cifar10_custom.resnet': (10,),


    'shakespeare.stacked_lstm': (80, 80, 256),
    'synthetic.mclr': (10,),
    'synthetic_iid.mclr': (10,),
    'celeba.cnn': (2,)
}

# =========================================================
# ARGUMENT PARSING
# =========================================================
def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, choices=OPTIMIZERS, default='fedavg')
    parser.add_argument('--dataset', type=str, choices=DATASETS, default='nist')
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--clients_per_round', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--mu', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--drop_percent', type=float, default=0.1)
    parser.add_argument('--clientsel_algo', type=str, default='random')
    parser.add_argument('--Ls0', type=int, default=2)
    parser.add_argument('--sim_metric', type=str, default='grad')
    parser.add_argument('--m_interval', type=int, default=1)

    parsed = vars(parser.parse_args())

    # -----------------------------------------------------
    # SEEDS
    # -----------------------------------------------------
    random.seed(1 + parsed['seed'])
    np.random.seed(12 + parsed['seed'])
    tf.set_random_seed(123 + parsed['seed'])

    # -----------------------------------------------------
    # MODEL IMPORT LOGIC (CRITICAL PART)
    # -----------------------------------------------------
    if parsed['dataset'].startswith("synthetic"):
        model_path = f"flearn.models.synthetic.{parsed['model']}"
        


    elif parsed['dataset'] == "mnist_custom":
        if parsed['model'] == "cnn":
            # CNN exists only under nist
            model_path = "flearn.models.nist.cnn"
        else:
            # MCLR exists under mnist
            model_path = "flearn.models.mnist.mclr"

    else:
        model_path = f"flearn.models.{parsed['dataset']}.{parsed['model']}"

    mod = importlib.import_module(model_path)
    learner = getattr(mod, 'Model')

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------
    opt_path = f"flearn.trainers.{parsed['optimizer']}"
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # -----------------------------------------------------
    # MODEL PARAMS (MATCH MODEL PATH)
    # -----------------------------------------------------
    # -----------------------------------------------------
    # MODEL PARAMS (ROBUST HANDLING)
    # -----------------------------------------------------
    if parsed['dataset'].startswith("synthetic"):
        parsed['model_params'] = MODEL_PARAMS["synthetic.mclr"]

    elif parsed['dataset'] == "mnist_custom" and parsed['model'] == "cnn":
        parsed['model_params'] = MODEL_PARAMS["nist.cnn"]

    elif parsed['dataset'] == "mnist_custom":
        parsed['model_params'] = MODEL_PARAMS["mnist_custom.mclr"]

    else:
        parsed['model_params'] = MODEL_PARAMS[f"{parsed['dataset']}.{parsed['model']}"]


    # -----------------------------------------------------
    # PRINT OPTIONS
    # -----------------------------------------------------
    print("\nArguments:")
    for k in sorted(parsed.keys()):
        print(f"\t{k}: {parsed[k]}")

    return parsed, learner, optimizer

# =========================================================
# MAIN
# =========================================================
def main():
    tf.logging.set_verbosity(tf.logging.WARN)

    start_time = time.time()

    options, learner, optimizer = read_options()

    train_path = os.path.join('data', options['dataset'], 'data', 'train')
    test_path = os.path.join('data', options['dataset'], 'data', 'test')


    dataset = read_data(train_path, test_path)

    server = optimizer(options, learner, dataset)
    server.train()

    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")

if __name__ == '__main__':
    main()

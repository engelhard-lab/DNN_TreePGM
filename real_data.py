#!/usr/bin/env python
#========================================================================
# Get calibration for real-world dataset
#========================================================================

import sys
import os
import time
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.keras import callbacks

import config
from datagen import make_breast_cancer, make_digit, make_mnist, covertype
from train import create_nn_model, create_pgm, model_eval
from synthetic import sampling

def train(dat_type, train_ds, test_ds, input_shape, layer_sizes, epochs, lr):

    '''
    Train a base NN with the same number of hidden layers and more hidden nodes in each layer.
    k is used for generating the same_model and training history files.
    Learning rate is set manually. No tuning or early stopping with only train dataset.
    is_split=False means training with whole dataset and without train/test split. The test data would be all possible input settings. 
    '''

    model = create_nn_model(input_shape, layer_sizes, lr)
    cp_path = config.model_path + f"cp_{dat_type}_bp_{epochs}_-4.ckpt"

    if os.path.exists(cp_path + ".index"):
        print("NN model has been trained.")
        model.load_weights(cp_path)
        pred_proba = model.predict(test_ds)

    else:
        cp_callback = callbacks.ModelCheckpoint(filepath=cp_path, save_weights_only=True, verbose=0)
        print("Start NN training.")
        start_time = time.time()
        train_hist = model.fit(train_ds, epochs=epochs, verbose=0, callbacks=[cp_callback])
        train_time = time.time() - start_time
        pred_proba = model.predict(test_ds)
 
        np.savez(config.hist_path + f"hist_{dat_type}_bp_{epochs}_-4.npz", time=train_time, hist=train_hist, proba=pred_proba)

    return model, pred_proba.reshape(-1)


def finetune(dat_type, train_ds, model_nn, layer_sizes, L, train_epochs, ft_epochs, lr):

    model_pgm, kernels = create_pgm(layer_sizes, lr, L, train_ds, model_nn=model_nn)
    network = [model_pgm.call(data) for data, target in train_ds]

    ft_losses, ft_accs, ft_aucs, ft_probas = [], [], [], []

    print(f"Start HMC_{L} Finetuning.")
    start_time = time.time()
    for epoch in range(ft_epochs):
        for bs, (data, target) in enumerate(train_ds):
            model_pgm.update_weights(data, network[bs], target)
            if L == -1: # Gibbs
                network = [model_pgm.gibbs_new_state(x, net, y) for (x, y), net in zip(train_ds, network)]
            else:
                network = [model_pgm.propose_new_state_hamiltonian(x, net, y, ker, is_update_kernel=False) for (x, y), net, ker in zip(train_ds, network, kernels)]

        ft_loss, ft_acc, ft_auc, ft_proba = model_eval(model_pgm, train_ds)
        print("Epoch %d/%d: - %.4fs/step - ft_loss: %.4f - ft_acc: %.4f - ft_auc: %.4f" 
              % (epoch + 1, ft_epochs, (time.time() - start_time) / (epoch + 1), ft_loss, ft_acc, ft_auc))
         
        ft_losses.append(ft_loss.numpy())       
        ft_accs.append(ft_acc)
        ft_aucs.append(ft_auc)
        ft_probas.append(ft_proba)

    ft_time = time.time() - start_time
    pgm_weights = model_pgm.get_weights()
    ft_hist = {"ft_acc": ft_accs, "ft_auc": ft_aucs, "ft_loss": ft_losses, "ft_proba": ft_probas}
    np.savez(config.model_path + f"ft_{dat_type}_{train_epochs}_-4_hmc_{L}_{ft_epochs}_-4.npz", weights=pgm_weights)
    np.savez(config.hist_path + f"histft_{dat_type}_{train_epochs}_-4_hmc_{L}_{ft_epochs}_-4.npz", time=ft_time, hist=ft_hist)

    return model_pgm

def main():

    random_seed = config.random_seed
    layer_sizes = config.layer_sizes
    lr_train = config.lr_train
    train_epochs = config.train_epochs
    lr_finetune = config.lr_finetune
    ft_epochs = config.ft_epochs

    dat_type = sys.argv[1]
    L = int(sys.argv[2])
    
    if dat_type == "wbc":
        input_shape, trn_ds, train_ds, val_ds, test_ds = make_breast_cancer(random_seed)
    elif dat_type == "digit":
        input_shape, trn_ds, train_ds, val_ds, test_ds = make_digit(random_seed)
    elif dat_type == "mnist":
        input_shape, trn_ds, test_ds = make_mnist(size=1000, seed=random_seed)
    elif dat_type == "cov1":
        input_shape, trn_ds, test_ds = covertype(label=1, size=1000, seed=random_seed)
    elif dat_type == "cov2":
        input_shape, trn_ds, test_ds = covertype(label=2, size=1000, seed=random_seed)

    model_nn, prob_nn = train(dat_type, trn_ds, test_ds, input_shape, layer_sizes, train_epochs, lr_train)
    
    # only finetune once to get the predicted pdf
    model_pgm = finetune(dat_type, trn_ds, model_nn, layer_sizes, L, train_epochs, ft_epochs, lr_finetune)
    all_sampled_probs, prob_pgm = sampling(model_pgm, test_ds)
    np.savez(config.res_path + f"res_{dat_type}_{train_epochs}_-4_hmc_{L}_{ft_epochs}_-4.npz", all_probs=all_sampled_probs, prob=prob_pgm)
    
if __name__ == "__main__":
    main()   
#!/usr/bin/env python
#========================================================================
# Training modules
#========================================================================

import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm
from tensorflow.keras import layers, optimizers, callbacks, Sequential
from sklearn.metrics import accuracy_score, roc_auc_score

import config
from model import StochasticMLP
from datagen import make_moon, make_digit, make_mnist, make_breast_cancer

def model_eval(model, dat, acc_only=False):
    
    y_true = np.concatenate([target for data, target in dat.as_numpy_iterator()])
    pred_probas = [model.predict_proba(data) for data, target in dat]
    pred_labels = [tf.cast(tfm.greater(proba, 0.5), tf.int32) for proba in pred_probas]
    acc = accuracy_score(y_true, np.concatenate(pred_labels))

    if acc_only: 
        return acc
    else:
        loss = 0.0
        for bs, (data, target) in enumerate(dat):
            loss += tf.reduce_mean(model.get_loss(data, target))
        loss /= (bs + 1) 
        auc = roc_auc_score(y_true, np.concatenate(pred_probas))
        return loss, acc, auc, pred_probas

def model_sampling(model, dat_test, is_gibbs=False):

    # save all results for different samples of hidden nodes and outputs to generate 95% CI
    y_true = np.concatenate([target for data, target in dat_test.as_numpy_iterator()])

    network = [model.call(data) for data, target in dat_test]

    if not is_gibbs:
        kernels = [model.generate_hmc_kernel(data, None) for data, target in dat_test]
        # sample hidden state with HMC
        for bs, (data, target) in enumerate(dat_test):
            network = [model.propose_new_state_hamiltonian(x, net, None, ker, is_update_kernel=False) 
                       for (x, y), net, ker in zip(dat_test, network, kernels)]
        
    # get output probability and loss
    total_loss = 0.0
    pred_probas = []
    for bs, (data, target) in enumerate(dat_test):
            
        logits = model.output_layer(network[bs][-1])
        pred_probas.append(tfm.sigmoid(logits))
            
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(target, tf.float32), logits=logits)
        total_loss += tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    avg_loss = total_loss.numpy() / (bs + 1)
    pred_labels = [tf.cast(tfm.greater(proba, 0.5), tf.int32) for proba in pred_probas]
    acc = accuracy_score(y_true, np.concatenate(pred_labels))
    auc = roc_auc_score(y_true, np.concatenate(pred_probas))

    return avg_loss, acc, auc, pred_probas

def create_nn_model(input_shape, sizes, lr):

    if isinstance(sizes, (int, float)):
        sizes = [sizes]
    
    input_layer = layers.InputLayer(input_shape=(input_shape,))
    dense_layers = [layers.Dense(size, activation="sigmoid") for size in sizes]
    output_layer = layers.Dense(1, activation="sigmoid")
    model = Sequential([input_layer] + dense_layers + [output_layer])
    
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy", "AUC"])
    print(model.metrics_names)

    return model

def create_pgm(layer_sizes, lr, L, dataset, model_nn=None):

    is_gibbs = (L == -1)
    model_pgm = StochasticMLP(hidden_layer_sizes=layer_sizes, n_outputs=1, lr=lr, is_gibbs=is_gibbs, L=L)
    network = [model_pgm.call(data) for data, target in dataset]

    if model_nn != None:
        for i, layer in enumerate(model_nn.layers[:-1]):
            model_pgm.fc_layers[i].set_weights(layer.get_weights())
        model_pgm.output_layer.set_weights(model_nn.layers[-1].get_weights())

    kernels = None
    if not is_gibbs:
        kernels = [model_pgm.generate_hmc_kernel(data, target) for data, target in dataset]

    return model_pgm, kernels
    
def bp(dat_train, dat_val, input_shape, size, epochs, lr):
    '''
    Standard Backpropogation training with hyperparameter tuning
    '''
    
    print("Start BP")
    model = create_nn_model(input_shape, size, lr)
    
    checkpoint_filepath = config.model_path + f"best_bp_size={size}_lr={lr}.pth"
    cp_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_auc',
        mode='max',
        save_best_only=True)
    
    start_time = time.time()
    train_hist = model.fit(dat_train, epochs=epochs, validation_data=dat_val, verbose=0, callbacks=[cp_callback])
    train_time = time.time() - start_time
    
    model.load_weights(checkpoint_filepath)
    pred_proba = model.predict(dat_val, verbose=0)
    
    return train_time, train_hist, pred_proba

def bp_test(dat_trn, dat_test, input_shape, size, epochs, lr, N_samples=-1):

    model_nn = create_nn_model(input_shape, size, lr)
    print(model_nn)

    # retrain the model with the whole train dataset
    hist = model_nn.fit(dat_trn, epochs=epochs, verbose=0)

    if N_samples == -1:
        pred_proba = model_nn.predict(dat_test, verbose=0)
        eval_test = model_nn.evaluate(dat_test, verbose=0)
        return pred_proba, eval_test
    else:
        # treat NN as a PGM and sample
        # construct a PGM with the same weight as the trained NN model
        # use gibbs way to sample
        model_pgm = StochasticMLP(hidden_layer_sizes=[size], n_outputs=1, lr=lr, is_gibbs=True, L=-1)
        network = [model_pgm.call(data) for data, target in dat_trn] # initial weights of the PGM

        for i, layer in enumerate(model_nn.layers[:-1]):
            model_pgm.fc_layers[i].set_weights(layer.get_weights())
        model_pgm.output_layer.set_weights(model_nn.layers[-1].get_weights())

        eval_tests = []
        pred_probas = []
        for i in range(N_samples):
            test_loss, test_acc, test_auc, pred_proba = model_sampling(model_pgm, dat_test, is_gibbs=True)
            eval_test = [test_loss, test_acc, test_auc]
            eval_tests.append(eval_test)
            pred_probas.append(np.concatenate(pred_proba))
        return pred_probas, eval_tests      

def mcmc(dat_train, dat_val, sizes, epochs, lr, burnin, is_gibbs, L):
    '''
    Gibbs/HMC training (w/wo Gaussian approximation)
    '''

    if is_gibbs:
        print("Start Gibbs")
    elif L == -1:
        print("Start HMC")
    else:
        print("Start HMC with Gaussian approximation")
        
    model = StochasticMLP(hidden_layer_sizes=sizes, n_outputs=1, lr=lr, is_gibbs=is_gibbs, L=L)
    network = [model.call(data) for data, target in dat_train]
    if not is_gibbs:
        kernels = [model.generate_hmc_kernel(data, target) for data, target in dat_train]
    
    # Burnin
    print("Start Burning")
    #burnin_losses = []
    for i in range(burnin):
        
        if(i % 100 == 0): print("Step %d" % i)

        res = []
        #burnin_loss = 0.0
        for bs, (data, target) in enumerate(dat_train):
            if is_gibbs:
                res.append(model.gibbs_new_state(data, network[bs], target))
            else:
                res.append(model.propose_new_state_hamiltonian(data, network[bs], target, kernels[bs]))

        if is_gibbs:
            network = res
        else:
            network, kernels = zip(*res)
    
    # Training
    print("Start Training")
    train_losses = []
    train_accs = []
    train_aucs = []
    train_probas = []
    val_losses = []
    val_accs = []
    val_aucs = []
    val_probas = []
    start_time = time.time()
    
    for epoch in range(epochs):

        # train
        for bs, (data, target) in enumerate(dat_train):
            #print(bs)
            model.update_weights(data, network[bs], target)
            if is_gibbs:
                network = [model.gibbs_new_state(x, net, y) for (x, y), net in zip(dat_train, network)]
            else:
                network = [model.propose_new_state_hamiltonian(x, net, y, ker, is_update_kernel = False) for (x, y), net, ker in zip(dat_train, network, kernels)]
            
        train_loss, train_acc, train_auc, train_proba = model_eval(model, dat_train)
        train_losses.append(train_loss)       
        train_accs.append(train_acc)
        train_aucs.append(train_auc)
        train_probas.append(train_proba)
        
        # validate
        val_loss, val_acc, val_auc, val_proba = model_eval(model, dat_val)
        val_losses.append(val_loss)       
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        val_probas.append(val_proba)
        
        print("Epoch %d/%d: - %.4fs/step - train_loss: %.4f - train_acc: %.4f - train_auc: %.4f - val_loss: %.4f - val_acc: %.4f - val_auc: %.4f" 
              % (epoch + 1, epochs, (time.time() - start_time) / (epoch + 1), train_loss, train_acc, train_auc, val_loss, val_acc, val_auc))

    train_time = time.time() - start_time
    train_hist = {"train_acc": train_accs, "train_auc": train_aucs, "train_loss": train_losses, "train_proba": train_probas,
                  "val_acc": val_accs, "val_auc": val_aucs, "val_loss": val_losses, "val_proba": val_probas}
    
    return train_time, train_hist, val_proba

def mcmc_test(dat_trn, dat_test, sizes, epochs, lr, burnin, is_gibbs, L, N_samples=-1):
        
    model = StochasticMLP(hidden_layer_sizes=sizes, n_outputs=1, lr=lr, is_gibbs=is_gibbs, L=L)
    network = [model.call(data) for data, target in dat_trn]
    if not is_gibbs:
        kernels = [model.generate_hmc_kernel(data, target) for data, target in dat_trn]

    for i in range(burnin):
        
        if(i % 100 == 0): print("Step %d" % i)
        res = []
        for bs, (data, target) in enumerate(dat_trn):
            if is_gibbs:
                res.append(model.gibbs_new_state(data, network[bs], target))
            else:
                res.append(model.propose_new_state_hamiltonian(data, network[bs], target, kernels[bs]))

    for epoch in range(epochs):
        # train
        for bs, (data, target) in enumerate(dat_trn):
            model.update_weights(data, network[bs], target)
            if is_gibbs:
                network = [model.gibbs_new_state(x, net, y) for (x, y), net in zip(dat_trn, network)]
            else:
                network = [model.propose_new_state_hamiltonian(x, net, y, ker, is_update_kernel = False) for (x, y), net, ker in zip(dat_trn, network, kernels)]

    if N_samples == -1:
        test_loss, test_acc, test_auc, pred_proba = model_eval(model, dat_test)
        eval_test = [test_loss, test_acc, test_auc]
        return pred_proba, eval_test
    else:
        eval_tests = []
        pred_probas = []
        for i in range(N_samples):
            test_loss, test_acc, test_auc, pred_proba = model_sampling(model, dat_test, is_gibbs=is_gibbs)
            eval_test = [test_loss, test_acc, test_auc]
            eval_tests.append(eval_test)
            pred_probas.append(np.concatenate(pred_proba))
        return pred_probas, eval_tests 

def main():
    
    epochs = 100
    burnin = 100
    seed = 42

    dataset = sys.argv[1]
    if dataset == "make_moon":
        input_shape, train_ds, val_ds = make_moon(size = 1000, noise = 0.3)
    elif dataset == "mnist":
        input_shape, train_ds, val_ds = make_mnist(size = 2000)
    elif dataset == "digit":
        input_shape, trn_ds, train_ds, val_ds, test_ds = make_digit(seed = seed)
    elif dataset == "breast_cancer":
        input_shape, trn_ds, train_ds, val_ds, test_ds = make_breast_cancer(seed = seed)

    model_type = sys.argv[2]
    if model_type not in ['bp', 'hmc', 'gibbs']:
        sys.exit("The model type must be one of the 'bp', 'hmc' or 'gibbs'.")

    lrs = [1e-4, 1e-3, 1e-2]
    sizes = [16, 32, 64, 128, 256, 512, 1024]

    best_val_auc = 0.0
    best_lr = 0
    best_size = 0

    hist_dir_path = config.hist_path + sys.argv[1] + "/"
    
    if model_type == 'bp':
        for size in sizes:
            for lr in lrs:
                # get result
                time, hist, pred_proba = bp(train_ds, val_ds, input_shape, size, epochs, lr)
                histname = hist_dir_path + sys.argv[1] + f"_hist_bp_adam_size={size}_lr={lr}.npz"
                np.savez(histname, time=time, hist=hist, proba=pred_proba)

                # maintain the best model
                max_val_auc = max(hist.history['val_auc'])
                print(size, lr, max_val_auc)
                if max_val_auc > best_val_auc:
                    best_val_auc = max_val_auc
                    best_lr = lr
                    best_size = size

        # retrain best model and then run it on the test dataset
        print(best_size, best_lr)
        pred_proba, eval_test = bp_test(trn_ds, test_ds, input_shape, best_size, epochs, best_lr)
        testres = hist_dir_path + sys.argv[1] + "_testres_bp_adam.npz"
        np.savez(testres, proba=pred_proba, res=eval_test)
                
    else:
        L = int(sys.argv[3])
        if model_type == 'hmc':
            is_gibbs = False
        else:
            is_gibbs = True
        for size in sizes:
            for lr in lrs:
                time, hist, pred_proba = mcmc(train_ds, val_ds, size, epochs, lr, burnin, is_gibbs, L)
                histname = hist_dir_path + sys.argv[1] + "_hist_" + sys.argv[2] + f"_L={L}_adam_size={size}_lr={lr}.npz"
                np.savez(histname, time=time, hist=hist, proba=pred_proba)

                max_val_auc = max(hist['val_auc'])
                print(size, lr, max_val_auc)
                if max_val_auc > best_val_auc:
                    best_val_auc = max_val_auc
                    best_lr = lr
                    best_size = size

        print(best_size, best_lr)     
        
if __name__ == "__main__":
    main()
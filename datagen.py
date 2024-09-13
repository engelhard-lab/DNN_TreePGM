#!/usr/bin/env python
#========================================================================
# Synthetic data generator and real data loader
#========================================================================

import numpy as np
import random
import tensorflow as tf
from pandas import read_csv
from tensorflow.keras.datasets import mnist
from sklearn.datasets import make_moons, load_digits, load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split

def covertype(label, size = 1000, seed = 42):

    X, y = fetch_covtype(return_X_y=True, random_state=seed, shuffle=True)
    y = (y == label).astype(int)
    
    test_size = 0.2
    Ntrain = int(size * (1 - test_size))
    Ntest = size - Ntrain

    x_train, y_train = X[:Ntrain], y[:Ntrain]
    x_test, y_test = X[Ntrain:size], y[Ntrain:size]
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    shape = x_train[0].shape[0]

    #print(sum(y_train==1), sum(y_test==1))

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    
    return shape, train_ds, test_ds

def make_moon(size = 1000, noise = 0.3, seed = 42):

    np.random.seed(seed)
    X, y = make_moons(size, noise = noise)

    # Split into test and training data
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=seed)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    shape = x_train[0].shape[0]

    #print(sum(y_train)/y_train.shape[0], sum(y_val)/y_val.shape[0])

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)

    return shape, train_ds, val_ds

def make_digit(seed = 42):

    x, y = load_digits(n_class=2, return_X_y=True)
    x_trn, x_test, y_trn, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_trn, y_trn, test_size=0.25, random_state=seed)

    y_trn = y_trn.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    shape = x_train[0].shape[0]

    trn_ds = tf.data.Dataset.from_tensor_slices((x_trn, y_trn)).batch(32)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    return shape, trn_ds, train_ds, val_ds, test_ds

def make_breast_cancer(seed = 42):

    x, y = load_breast_cancer(return_X_y=True)
    x_trn, x_test, y_trn, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_trn, y_trn, test_size=0.25, random_state=seed)

    y_trn = y_trn.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    shape = x_train[0].shape[0]

    trn_ds = tf.data.Dataset.from_tensor_slices((x_trn, y_trn)).batch(32)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    return shape, trn_ds, train_ds, val_ds, test_ds

def make_mnist(size = -1, seed = 42):
    
    # Load MNIST
    (x_train, y_train), (x_val, y_val) = mnist.load_data()

    # Select binary data
    classes = [0,1]
    x_train = [x.reshape(-1) for x, y in zip(x_train, y_train) if y in classes]
    y_train = [y.reshape(-1) for y in y_train if y in classes]
    x_val = [x.reshape(-1) for x, y in zip(x_val, y_val) if y in classes]
    y_val = [y.reshape(-1) for y in y_val if y in classes]
    shape = x_train[0].shape[0]

    #print('There are', len(x_train), 'training images.')
    #print('There are', len(x_val), 'validation images.')

    x_train = [(x - 127.5) / 255 for x in x_train]
    x_val = [(x - 127.5) / 255 for x in x_val]

    random.seed(seed)
    # randomly choose a subset for train and validation
    if size != -1:
        # train
        N_train = int(size * 0.8)
        xy_train_0 = [(x, y) for x, y in zip(x_train, y_train) if y == 0]
        xy_train_1 = [(x, y) for x, y in zip(x_train, y_train) if y == 1]
        print(len(xy_train_0), len(xy_train_1))
        xy_train_sub = random.sample(xy_train_0, N_train // 2) + random.sample(xy_train_1, N_train // 2)
        x_train_sub = [x for x, y in xy_train_sub]
        y_train_sub = [y for x, y in xy_train_sub]

        # val
        N_val = int(size * 0.2)
        xy_val_0 = [(x, y) for x, y in zip(x_val, y_val) if y == 0]
        xy_val_1 = [(x, y) for x, y in zip(x_val, y_val) if y == 1]
        print(len(xy_val_0), len(xy_val_1))
        xy_val_sub = random.sample(xy_val_0, N_val // 2) + random.sample(xy_val_1, N_val // 2)
        x_val_sub = [x for x, y in xy_val_sub]
        y_val_sub = [y for x, y in xy_val_sub]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train_sub, y_train_sub)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val_sub, y_val_sub)).batch(32)

    return shape, train_ds, val_ds

def synthetic_bn(k, seed = 42): # k is the index of the synthetic datasets

    # load data
    data_dir = "/hpc/group/pagelab/bl222/dnn-mcmc/experiments/normal_approx/synthetic_data/bn/"
    data = np.load(f"{data_dir}/bn_synthetic_{k}.npz", allow_pickle=True)
    x, y = data['x'], data['y']
    dim_x = x.shape[1]

    # return datasets
    #x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

    '''
    # test dataset is all possible settings of the input
    x_test_bin = [bin(i)[2:].zfill(dim_x) for i in range(2**dim_x)]
    x_test = np.array([list(map(int, list(binary))) for binary in x_test_bin])
    #print("All input settings include ", x_test)
    '''

    #trn_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    #val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
    #test_ds = tf.data.Dataset.from_tensor_slices(x_test).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    #return dim_x, trn_ds, train_ds, val_ds, test_ds
    return dim_x, train_ds, test_ds

def synthetic_mn(k, seed = 42):

    data_dir = "/hpc/group/pagelab/bl222/dnn-mcmc/experiments/normal_approx/synthetic_data/mn"
    df = read_csv(f"{data_dir}/mn_simulated_{k}.csv", header=None)
    x, y = df.values[:, :-2], df.values[:, -2:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    dim_x = x.shape[1]

    # For train dataset, we use labels (0/1) as the true output
    y_train = y_train[:,0].reshape(-1, 1)

    # For test dataset, we use the prob as the true output
    y_test = y_test[:,1]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    return dim_x, train_ds, test_ds
        
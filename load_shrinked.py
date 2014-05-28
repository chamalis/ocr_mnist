import numpy as np
import random
import sys

def load_dataset():
        
    '''Read shrinked dataset'''

    npztrain = np.load('MNIST/shrinked/train.npz')
    npzvalid = np.load('MNIST/shrinked/valid.npz')
    npztest = np.load('MNIST/shrinked/test.npz')

    train = npztrain[npztrain.files[0]]  # Nx14x14 , numpy.ndarray, (N=50000)
    valid = npzvalid[npzvalid.files[0]]
    test = npztest[npztest.files[0]]

    #targets - retrieved imediately in array format - :Nx1  eg 50000,1
    traint = np.load('MNIST/shrinked/train_targets.npy')
    validt = np.load('MNIST/shrinked/valid_targets.npy')
    testt = np.load('MNIST/shrinked/test_targets.npy')

    #shuffle test set 
    order = range(np.shape(test)[0])
    random.shuffle(order)
    test = test[order][:][:]
    testt = testt[order][:]

    #shuffle training - optional, its eitherway shuffled after iteration in train func
    order = range(np.shape(train)[0])
    random.shuffle(order)
    train = train[order][:]
    traint = traint[order][:]

    return train, valid, test, traint, validt, testt

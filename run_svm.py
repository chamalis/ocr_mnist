import numpy as np

from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

import time, random, math
from load_shrinked import load_dataset

#import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d.axes3d as p3
#import matplotlib.dates as dates

PERCENT_DATASET_USED = 23

def print_time_elapsed(start):
    end = time.time()
    seconds = end-start
    minutes = math.floor(seconds / 60)
    secs = seconds % 60
    print 'time elapsed: ' +str(minutes) + 'min ' +str(secs) +'s' + '\n\n'


train_set, valid_set, test_set, traint_init, validt_init, testt = load_dataset()

#Concatanate train, valid sets
train_set = np.concatenate((train_set, valid_set), axis=0)
traint = np.concatenate((traint_init, validt_init), axis=0) 

#Reshape X from NX14x14 to NX196 , Y remains Nx1
train = np.zeros((np.shape(train_set)[0], 196,))
for i in range(0, np.shape(train)[0]):
    train[i] = train_set[i].reshape(196,)
test = np.zeros((np.shape(test_set)[0], 196,))
for i in range(0, np.shape(test)[0]):
    test[i] = test_set[i].reshape(196,)

print np.shape(train), np.shape(traint)
print np.shape(test), np.shape(testt)

# Normalization to -1 1
nmean =  train.mean()
nmax = train.max()
train[:]= (train[:]-nmean) / nmax
test[:]= (test[:]-nmean) / nmax


#Choose a subset or the whole set 
train= train[: PERCENT_DATASET_USED *(np.shape(train)[0]) / 100]
traint = traint[: PERCENT_DATASET_USED *(np.shape(traint)[0]) / 100]
print np.shape(train), np.shape(traint)


'''This way we decide for the best parameters (starting from a wider range and 
   approaching more (closely)-precisely the range looking for the optimal values
   Search for RBF kernel '''

print 'Performing (grid search) for the optimal parameters of the rbf kernel'
start = time.time()
C_range = 2.0 ** np.arange(-2, 2.5, 0.5)
gamma_range = 2.0 ** np.arange(-10, -2, 0.5)
param_rbf_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(y=traint, n_folds=3) #cross validation 
grid_rbf = GridSearchCV(SVC(kernel='rbf'), param_grid=param_rbf_grid, cv=cv)
grid_rbf.fit(train, traint)
print ("The best classifier is: " , grid_rbf.best_estimator_)
print ('score ', grid_rbf.score(test,testt))
print_time_elapsed(start)


'''Same thing for the polynomial kernel'''

print 'Performing (grid search) for the optimal parameters of the polynomial kernel'
start = time.time()
C_range = 2.0 ** np.arange(-3, 2, 0.4)
gamma_range = 2.0 ** np.arange(-7,-2,0.35)
r_range = 2.0 ** np.arange(-6,-1,0.35)
degree_range = range(3,7)
cv = StratifiedKFold(y=traint, n_folds=3) #cross validation 
param_pol_grid = dict(gamma=gamma_range, C=C_range, degree=degree_range, coef0=r_range)
grid_pol = GridSearchCV(SVC(kernel='poly'), param_grid=param_pol_grid, cv=cv)
grid_pol.fit(train, traint)
print ("The best classifier is: " , grid_pol.best_estimator_)
print ('score ', grid_pol.score(test,testt))
print_time_elapsed(start)


#applying best parameters

print 'training RBF - SVM ... '
start = time.time()
# Initialise the model, for the best parameters found
clf = SVC(kernel="rbf", C=2.0, gamma=.0625)
clf.fit(train, traint)
print clf.score(test,testt)
print_time_elapsed(start)


print 'training SVM - poly ... '
start = time.time()
# Initialise the model, for the best parameters found
clf = SVC(kernel="poly", degree=3, C=0.35, coef0=0.125, gamma=0.0625)
clf.fit(train, traint)
print clf.score(test,testt)
print_time_elapsed(start)


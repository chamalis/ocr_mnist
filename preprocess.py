import cPickle, gzip
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import copy
import scipy.ndimage 
from skimage.transform import resize
import sys, os

DATASET_PATH = 'MNIST'
DATASET_FILE = os.path.join(DATASET_PATH, 'mnist.pkl.gz')

#If dataset doesn't exist locally, download it first
if os.path.exists(DATASET_FILE) == False:
    if os.path.exists(DATASET_PATH) == False:
        print 'creating dir: ', DATASET_PATH
        os.mkdir(DATASET_PATH)   
    import urllib
    print 'Downloading MNIST dataset ... '
    urllib.urlretrieve ("http://deeplearning.net/data/mnist/mnist.pkl.gz", DATASET_FILE) 

# Load the dataset
f = gzip.open(DATASET_FILE, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#Create directory to save the shrinked (preprocessed) dataset
SHRINKED_PATH = os.path.join(DATASET_PATH, 'shrinked')
if os.path.exists(SHRINKED_PATH) == False:
    print 'Creating dir ', SHRINKED_PATH
    os.mkdir(SHRINKED_PATH)
    
print 'shrinking training set'
arrays = []
train = np.array(train_set[0][:])
traint = np.zeros((np.shape(train)[0],10))
for i in range (0, np.shape(train)[0]):

    pic = train_set[0][i]
    
    a = copy.deepcopy(pic)  #copy not reference
 
    aa = a.reshape(28,28)  # from 784x1 
    
    #remove padding (actual image: 20x20)
    aa = aa[4:24, 4:24]

    im = Image.fromarray(aa*255)
#   im.show()
    cc = im.resize((14,14))
#   cc.show()
    
    arr = np.array(cc)
    #arr = arr/arr.max()
    arrays.append(arr)

#Save the shrinked train set
filename = os.path.join(SHRINKED_PATH, 'train')
np.savez(filename, arrays)
filename_targets = os.path.join(SHRINKED_PATH, 'train_targets')
np.save(filename_targets, train_set[1])


print 'shrinking validation set'
arrays = []
valid = np.array(valid_set[0][:])
validt = np.zeros((np.shape(valid)[0],10))
for i in range (0, np.shape(valid)[0]):

    pic = valid_set[0][i]
    a = copy.deepcopy(pic)  #copy not reference
 
    aa = a.reshape(28,28)
    aa = aa[4:24, 4:24]

    im = Image.fromarray(aa*255)
    cc = im.resize((14,14))
    
    arr = np.array(cc)
    #arr = arr/arr.max()
    arrays.append(arr)

#Save the shrinked valid set
filename = os.path.join(SHRINKED_PATH, 'valid')
np.savez(filename, arrays)
filename_targets = os.path.join(SHRINKED_PATH, 'valid_targets')
np.save(filename_targets, valid_set[1])


print 'shriking test set'
arrays = []
test = np.array(test_set[0][:])
testt = np.zeros((np.shape(test)[0],10))
for i in range (0, np.shape(test)[0]):
    
    pic = test_set[0][i]
    a = copy.deepcopy(pic)  #copy not reference
 
    aa = a.reshape(28,28)
    aa = aa[4:24, 4:24]  #remove padding!

    im = Image.fromarray(aa*255)
    cc = im.resize((14,14))
    
    arr = np.array(cc)
    #arr = arr/arr.max()
    arrays.append(arr)

#Save the shrinked test set
filename = os.path.join(SHRINKED_PATH, 'test')
np.savez(filename, arrays)
filename_targets = os.path.join(SHRINKED_PATH, 'test_targets')
np.save(filename_targets, test_set[1])

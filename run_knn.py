from pylab import *  #ALWAYS IMPORT PYLAB BEFORE THE OTHER LIBRARIES
import subprocess, sys, time, random, math
from numpy import *
from knn import *
from matplotlib.pyplot import *
from load_shrinked import load_dataset

''' Fourier descriptors not working as it should do, so dont use it
    (at least not until a fixed implementation is submitted) '''

percent_dataset_usage = 1
feature = 'input_space'    #'input_space' or 'fds' - Use input_space
ks = [1,3]                 #k-values (classifiers) - multiple values

'''Read shrinked dataset'''
train, valid, test, traint, validt, testt = load_dataset()

#concatenate train, valid sets
train = np.concatenate((train, valid), axis=0)
traint = np.concatenate((traint,validt), axis=0)

#Sampling a subset of dataset
train = train[:percent_dataset_usage*60000/100]
traint = traint[:percent_dataset_usage*60000/100]     

    
start = time.time()

if feature == 'fds':
    
    '''train set'''
    contours_train=[]
    for i in range(0,shape(train)[0]):
        pic = 255*train[i]
        contours_train.append(contour(pic, levels=[35], colors='black', origin='image'))

    ffts_train=[]
    for i in range(0,shape(train)[0]):
        c = contours_train[i]
        segs = c.allsegs[0][0]
        oneDsegs = segs.ravel()
        ffts_train.append(fft.fft(oneDsegs, 100))
        
    '''test set'''
    contours_test=[]
    for i in range(0,shape(test)[0]):
        pic = 255*test[i]
        contours_test.append(contour(pic, levels=[35], colors='black', origin='image'))

    ffts_test=[]
    for i in range(0,shape(test)[0]):
        c = contours_test[i]
        segs = c.allsegs[0][0]
        oneDsegs = segs.ravel()
        ffts_test.append(fft.fft(oneDsegs, 100))
        
    print np.shape(ffts_train)



# Convert our data set into an easy format to use.
# This is a list of (x, y) pairs. x is an image, y is a label.    
dataset = []
if feature == 'input_space':
    for i in range(0, len(train)):
        dataset.append((train[i, :, :], traint[i]))
elif feature == 'fds':
    for i in range(0, len(ffts_train)):
        dataset.append((ffts_train[i], traint[i]))

    
# Create a classifier for various values of k.
classifiers = [kNN(dataset, k, feature) for k in ks] 


def predict_test(classifier, test):
    """Compute the prediction for every element of the test set."""
    predictions = [classifier.classify(test[i]) 
                   for i in range(0, len(test))]
    return predictions

print 'Searching the dataset for the neighbors...'
predictions = []
exec_times = []
for classifier in classifiers:
    startk = time.time()
    if feature == 'input_space':
        predictions.append(predict_test(classifier, test))
    elif feature == 'fds':
        predictions.append(predict_test(classifier, ffts_test))
    endk = time.time()
    exec_times.append( (endk-startk)/60 )


def evaluate_prediction(predictions, answers):
    """Compute how many were identical in the answers and predictions,
    and divide this by the number of predictions to get a percentage."""
    correct = sum(asarray(predictions) == asarray(answers))
    total = float(prod(answers.shape))
    return correct / total

labels = asarray(testt)
accuracies = [evaluate_prediction(pred, labels) for pred in predictions]
print shape(accuracies), type(accuracies), len(accuracies)

# Print results
for i in range(0, len(classifiers)):
    print 'k='+str(ks[i]) + ': '+ str(100*accuracies[i]) + '% accuracy'+', exec time= ' + str(exec_times[i]) + 'min'

# Draw the figure.
fig = figure()
plot(ks, accuracies, 'ro', figure=fig)
fig.suptitle("Nearest Neighbor Classifier Accuracies")
fig.axes[0].set_xlabel("k (# of neighbors considered)")
fig.axes[0].set_ylabel("accuracy (% correct)");
fig.axes[0].axis([0, max(ks) + 1, 0, 1]);
show()

end = time.time()
seconds = end-start
minutes = math.floor(seconds / 60)
secs = seconds % 60
print 'time elapsed: ' +str(minutes) + 'min ' +str(secs) +'s'

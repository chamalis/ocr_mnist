import cPickle, gzip, signal, argparse, sys, time, copy, random, math
import numpy as np
#import Image
from load_shrinked import load_dataset

# Load the dataset

train_set, valid_set, test_set, traint_init, validt_init, testt_init = load_dataset()

#print traint_init[0:5], testt_init[0:5]  #5,0,4,1,9  7,2,1,0,4

''' 1-N encoding '''
traint = np.zeros((np.shape(traint_init)[0],10))   # :Mx10 (eg 10000 x10) numpy array
for i in range (0, np.shape(traint_init)[0]):
    traint[i][traint_init[i]]=1

validt = np.zeros((np.shape(validt_init)[0],10))   # :Mx10 (eg 10000 x10) numpy array
for i in range (0, np.shape(validt_init)[0]):
    validt[i][validt_init[i]]=1
    
testt = np.zeros((np.shape(testt_init)[0],10))   # :Mx10 (eg 10000 x10) numpy array
for i in range (0, np.shape(testt_init)[0]):
    testt[i][testt_init[i]]=1

''' Reshape data from Mx14x14 to Mx196 '''
train = np.zeros((np.shape(train_set)[0], 196,))
for i in range(0, np.shape(train_set)[0]):
    train[i] = train_set[i].reshape(196,)

valid = np.zeros((np.shape(valid_set)[0], 196,))
for i in range(0, np.shape(valid_set)[0]):
    valid[i] = valid_set[i].reshape(196,)

test = np.zeros((np.shape(test_set)[0], 196,))
for i in range(0, np.shape(test_set)[0]):
    test[i] = test_set[i].reshape(196,)

#print np.shape(train)
#print np.shape(test)
#print np.shape(valid)
#print type(train[0])
#print np.shape(train[0])
#print np.shape(traint)
#print np.shape(validt)
#print np.shape(testt)


def normalize_data(train, test, valid=None):
    '''Normalization to -1 1'''
    nmean =  train.mean() #mean intensity of all arrays
    nmax = train.max()    #max intensity of all arrays
    #print nmax
    train[:]=train[:]-nmean
    train[:]=train[:]/nmax
    test[:]=test[:]-nmean
    test[:]=test[:]/nmax
    if valid != None:
        valid[:]=valid[:]-nmean
        valid[:]=valid[:]/nmax
    return train, test, valid
    
def sample_dataset(train, traint, percent_dataset_usage):
    '''Sample a subset of the dataset''' 
    train = train  [0 : (np.shape(train)[0] * percent_dataset_usage / 100) ]   
    traint = traint[0 : (np.shape(traint)[0] * percent_dataset_usage/100)]
    return train, traint
    
def save_NN_instance(filename):
    print 'saving neural network .. in ' + filename
    f = file(filename, 'wb')
    cPickle.dump(net, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    return

def print_NN_params():
    print 'number of layers ' + str(nlayers)
    print 'number of 1st layer nodes '+ str(nhidden)
    print 'number of 2nd layer nodes '+ str(nhiddeno)
    print 'Using ' + str(percent_dataset_usage) + '% of the dataset'
    print 'Training data dimensions: '+ str(np.shape(train))
    return

def print_time_elapsed(start):
    end = time.time()
    seconds = end-start
    minutes = math.floor(seconds / 60)
    secs = seconds % 60
    print 'time elapsed: ' +str(minutes) + 'min ' +str(secs) +'s'
    return

def signal_handler(signal, frame):
    ''' Ctrl-C handler. Save the ANN before exiting'''        
    net.confmat(test,testt)   #autopsy report
    print_NN_params()
    print_time_elapsed(start)
    filename = 'instances/NN_' +str(percent_dataset_usage) +'perc_'+ str(nhidden) +'_'+ str(nhiddeno) +'_stopped.save'
    save_NN_instance(filename)    
    exit()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='coming soon')    
    parser.add_argument('-t', '--train',  
                        help='''train function to use Back-propagation or Resilient BackPropagation (B/R)
                                ,default=B''', default='B')
    parser.add_argument('-l', '--layers',  
                        help='number of layers (1 or 2 only implemented)', default=1)
    parser.add_argument('-n1', '--nodes1',  
                        help='number of 1st layer nodes, (keep it low or go vacation)', default=50)
    parser.add_argument('-n2', '--nodes2',  
                        help='number of 2nd layer nodes, (keep it low or go vacation)', default=50)
    parser.add_argument('-du', '--dusage',  
                        help='percentage of dataset used (default=10), enter 100 for full dataset', default=10)
    args = parser.parse_args()
    nhidden = int(args.nodes1)
    nlayers = int(args.layers)
    percent_dataset_usage = int(args.dusage)
    train_func = args.train
    if nlayers > 1:
        nhiddeno = int(args.nodes2)
    else:
        nhiddeno = 0
    
    train, traint = sample_dataset(train, traint, percent_dataset_usage)
    train, test, valid = normalize_data(train, test, valid)
    
    print_NN_params() #remind us what architecture was tested
    start = time.time()

    if train_func == 'B':
        '''BackPropagation makes use of mlp.py and mlp2.py, codes for 1 and 2 layer Networks respectively,
           written by Stephen Marsland (check modules for ref) '''
        
        import mlp, mlp2 #, mlp_threaded - Not implemented yet
        signal.signal(signal.SIGINT, signal_handler)  #register the signal handler

        #Build the network 
        if nlayers > 1:
            net = mlp2.mlp(train,traint, nhidden, nhiddeno, outtype='softmax')
        elif nlayers == 1:
            net = mlp.mlp(train,traint, nhidden, outtype='softmax')
        
        print 'training mlp...'
        net.mlptrain(train,traint, 0.1, 700*percent_dataset_usage) #train some iterations before calling earlystopping!!
        net.earlystopping(train,traint,valid,validt,0.1)  #train until validation error start increasing
        net.confmat(test,testt)   #autopsy report 
    
    elif train_func == 'R':  
        ''' Resilient Propagation makes use of pyBrain framework (must be installed).
            Significantly faster than Bprop!! '''
        #Signal handler (Ctrl-C) not implemented yet for Rprop... meaning if stopped its lost
        
        # Train the network
        from pybrain.datasets            import ClassificationDataSet
        from pybrain.utilities           import percentError
        from pybrain.tools.shortcuts     import buildNetwork
        from pybrain.supervised.trainers import BackpropTrainer
        from pybrain.supervised.trainers import RPropMinusTrainer
        from pybrain.structure.modules   import SoftmaxLayer

        #net = mlp2.mlp(train,traint, nhidden, nhiddeno, outtype='softmax')
        #1-N output encoding , N=10 
        trndata = ClassificationDataSet(np.shape(train)[1], 10, nb_classes=10) 
        for i in xrange(np.shape(train)[0]):
            trndata.addSample(train[i], traint[i])
        validata = ClassificationDataSet(np.shape(valid)[1], 10, nb_classes=10) 
        for i in xrange(np.shape(valid)[0]):
            trndata.addSample(valid[i], validt[i])
        testdata = ClassificationDataSet(np.shape(test)[1], 10, nb_classes=10)
        for i in xrange(np.shape(test)[0]):
            testdata.addSample(test[i], testt[i])
        
        #Build the network 
        if nlayers > 1:
            net = buildNetwork(trndata.indim, nhidden, nhiddeno, trndata.outdim, outclass=SoftmaxLayer )
        else:
            net = buildNetwork(trndata.indim, nhidden, trndata.outdim, outclass=SoftmaxLayer )
        #construct the trainer object
        #We can also train Bprop using pybrain using the same argumets as below: trainer = BackpropTrainer(...)
        trainer = RPropMinusTrainer(net, dataset=trndata, momentum=0.9, verbose=True, weightdecay=0.01, learningrate=0.1)
        #train and test
        trainer.trainUntilConvergence(maxEpochs=percent_dataset_usage*300)#,trainingData=trndata,validationData = validata)
        trainer.testOnData(verbose=True, dataset=testdata)
        
        
    print_NN_params() #remind us what architecture was tested
    print_time_elapsed(start)  #print training time
    filename = 'instances/NN_' +str(percent_dataset_usage) +'perc_'+ str(nhidden) + '_' +str(nhiddeno) +'.save'
    save_NN_instance(filename) #save trained object to disk
    
    

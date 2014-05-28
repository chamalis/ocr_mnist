# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008
# Slightly modified by Me (chefarov@gmail.com), 2014 

from numpy import *
import sys

class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic'):
        """ Constructor """
        
        # Set up network size
        self.nin = shape(inputs)[1]
        self.nout = shape(targets)[1]
        self.ndata = shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network
        self.weights1 = (random.rand(self.nin+1,self.nhidden)-0.5)*2/sqrt(self.nin)
        self.weights2 = (random.rand(self.nhidden+1,self.nout)-0.5)*2/sqrt(self.nhidden)

        self.k = 0.001  #avoid division by zero in softmax (bug in Marsland's initial code)


    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
    
        valid = concatenate((valid,-ones((shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*sum((validtargets-validout)**2)
            
        #print "Stopped", new_val_error,old_val_error1, old_val_error2
        return new_val_error
        
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = concatenate((inputs,-ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
    
        updatew1 = zeros((shape(self.weights1)))
        updatew2 = zeros((shape(self.weights2)))
                      
        for n in range(niterations):
    
            self.outputs = self.mlpfwd(inputs)

            error = 0.5*sum((targets-self.outputs)**2)
            if (mod(n,20)==0):
                print "Iteration: ",n, " Error: ",error    

            # Different types of output neurons
            if self.outtype == 'linear':
                deltao = (targets-self.outputs)/self.ndata
            elif self.outtype == 'logistic':
                deltao = (targets-self.outputs)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                #deltao = (targets-self.outputs)*self.outputs/self.ndata
                deltao = (targets-self.outputs)/self.ndata
            else:
                print "error"
            
            deltah = self.hidden*(1.0-self.hidden)*(dot(deltao,transpose(self.weights2)))

            updatew1 = eta*(dot(transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(dot(transpose(self.hidden),deltao)) + self.momentum*updatew2
            self.weights1 += updatew1
            self.weights2 += updatew2
                
            # Randomise order of inputs
            random.shuffle(change)
            inputs = inputs[change,:]
            targets = targets[change,:]
            
    def mlpfwd(self,inputs):
        
        """ Run the network forward """
        self.hidden = dot(inputs,self.weights1)
        self.hidden = 1.0/(1.0+exp(-self.beta*self.hidden))
        self.hidden = concatenate((self.hidden,-ones((shape(inputs)[0],1))),axis=1)

        outputs = dot(self.hidden,self.weights2);
        # Different types of output neurons
        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = sum(exp(outputs),axis=1)*ones((1,shape(outputs)[0]))
            return transpose(transpose(exp(outputs))/(self.k+normalisers))
        else:
            print "error"

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = concatenate((inputs,-ones((shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        #print 'outputs:\n'+ str(outputs) 
        
        nclasses = shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = argmax(outputs,1)
            print 'outputs:\n'+str(outputs) 
            targets = argmax(targets,1)
            print 'targets:\n'+str(targets)

        cm = zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = sum(where(outputs==i,1,0)*where(targets==j,1,0))

        print "Confusion matrix is:"
        set_printoptions(precision=2)
        print  cm
        print "Percentage Correct: ",trace(cm)/sum(cm)*100


    '''chefarov 2014  - optional functions '''
    
    def testnet(self, inputs, targets):
        ''' for already trained network. test on another test set'''

        print shape(inputs), shape(self.weights1)
        self.nin = shape(inputs)[1]
        self.nout = shape(targets)[1]
        self.ndata = shape(inputs)[0]
        sys.stdin.read(1)
        inputs = concatenate((inputs,-ones((shape(inputs)[0],1))),axis=1)        
        out = self.mlpfwd(inputs)
        test_error = 0.5*sum((targets-out)**2)
        return test_error, out       

    def produce_output(self, input):
        '''Just calls mlpfwd, making sure that our input is well formed'''
        
        #print 'inside produce_output'
        #print shape(input)
        self.nin = 1       #that must be 1
        bias = zeros((1,))
        input = concatenate((input,bias),axis=0) #not axis=1 -> batch training 
        #print shape(input), shape(self.weights1)
        input = input.reshape(self.nin, shape(self.weights1)[0])
#        print self.nin
#        print shape(input)
#        print shape(self.weights1)     #785x100
        out = self.mlpfwd(input)
        return out
        
    

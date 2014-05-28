# Based on Andrew Gibiansky's code found at
# http://andrew.gibiansky.com/blog/machine-learning/k-nearest-neighbors-simplest-machine-learning/

# Heavily modified by chefarov@gmail.com, 2014

from collections import defaultdict
import sys
from numpy import *

class kNN(object):
    
    def __init__(self, dataset, k, feature):
        """Create a new nearest neighbor classifier.

        dataset - a list of data points. Each data point is an (x, y) pair,
                  where x is the input and y is the label.
        k - the number of neighbors to search for."""
        # Note how we don't have to do any initialization!
        # Once we have a dataset, we can immediately get predictions on new values.
        self.dataset = dataset
        self.k = k
        self.feature = feature

    def distance(self, p1, p2):
        #print shape(p1), shape(p2)
        #print type(p1), type(p2)
        if self.feature == 'input_space':
            return self.euclidean_distance(p1, p2)
            #return self.euclidean_distance(p1, p2)
        elif self.feature == 'fds':
            return self.fds_distance(p1, p2)
     
    def fds_distance(self, p1, p2):
        ''' Fourier descriptors Euklidean distance '''
        p1 = p1 / p1[1]
        p2 = p2 / p2[1]
        p1_abs = absolute(p1)
        p2_abs = absolute(p2)
        #print shape(p1_abs)
        #sys.stdin.read(1)
        dist = linalg.norm(p1_abs - p2_abs) #Euklidean distance       
        return dist
        
    def euclidean_distance(self, img1, img2):
    # Since we're using NumPy arrays, all our operations are automatically vectorized.
        distance = sum((img1[:][:] - img2[:][:]) ** 2)
#        print distance
#        print shape(distance)
        return distance
        
    def L3(self, img1, img2):
    # Since we're using NumPy arrays, all our operations are automatically vectorized.
        distance = (sum((img1[:][:] - img2[:][:]) ** 3) ** (1/3))
#        print distance
#        print shape(distance)
        return distance
        
    def get_majority(self, votes):
        '''For convenience, we're going to use a defaultdict.
          This is just a dictionary where values are initialized to zero
          if they don't exist. '''
        counter = defaultdict(int)
        for vote in votes:
            # If this weren't a defaultdict, this would error on new vote values.
            counter[vote] += 1
    
        # Find out who was the majority.
        majority_count = max(counter.values())
        for key, value in counter.items():
            if value == majority_count:
                return key

    def classify(self, point):
        # We have to copy the data set list, because once we've located the best
        # candidate from it, we don't want to see that candidate again, so we'll delete it.
        candidates = self.dataset[:]
        
        # Loop until we've gotten all the neighbors we want.
        neighbors = []
        while len(neighbors) < self.k:
            # Compute distances to every candidate.
            distances = [self.distance(x[0], point) for x in candidates]   #list of arrays INSTEAD of list of floats
            #print distances, type(distances)
            #sys.stdin.read(1)
            # Find the minimum distance neighbor.
            best_distance = min(distances)
            index = distances.index(best_distance)
            neighbors.append(candidates[index])

            # Remove the neighbor from the candidates list.
            del candidates[index]
        
        # Predict by averaging the closets k elements.
        prediction = self.get_majority([value[1] for value in neighbors])
        return prediction
            
  

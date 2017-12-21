# -*- coding: utf-8 -*-
"""
Spyder Editor
print 'i love you'
This is a temporary script file.
"""
from __future__ import print_function
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt



plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#load the raw CIFAR-10 data
cifar10_dir='cs231n/datasets/cifar-10-batches-py'
X_train,y_train,X_test,y_test = load_CIFAR10(cifar10_dir)
#cheak the shape of the train and test datasets
print('Training data shape: ',X_train.shape)
print('Training labels shape: ',y_train.shape)
print('Test data shape: ',X_test.shape)
print('Test labels shape:',y_test.shape)

#visualize some examples from the datasets
# We show a few examples of training from each class
classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_classes = len(classes)
samples_per_class = 7

for y,cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace= False)
    for i,idx in enumerate(idxs):
        plt_idx = i*num_classes +y+1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i ==0:
            plt.title(cls)
plt.show()

# subsample the training data for more efficient code execution
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0],-1))
X_test = np.reshape (X_test, (X_test.shape[0],-1))
print('The shape of the new selected training dataset:', X_train.shape,X_test.shape)

from cs231n.classifiers import KNearestNeighbor
#create a kNN classifier instance
#The classifier simply remember the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train) 

# open cs231n/classifiers /k_nearest_neighbor.py and implement
#compute distances by two loops

dists = classifier.compute_distances_two_loops(X_test)
print(dists.shape)

# We can visualize the distance matrix: each row is a single test example and 
#its distance to training examples 
plt.imshow(dists,interpolation = 'none') 
plt.show()

# Now run the prediction fuction predict_labels and run the code
# first try k=1
y_test_pred = classifier.predict_labels(dists, k =1)
# compute and print the fraction of correctly predicted examples

num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct)/num_test
print('Got %d /%d correct => accuracy: %f'%(num_correct,num_test,accuracy))

#secondly set k=5
y_test_pred = classifier.predict_labels(dists,k=5)
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct)/num_test
print('Using k=5, Got %d /%d correct => accuracy: %f'%(num_correct,num_test,accuracy))

#Now lets speed up distance matrix computation by using partial vectorization with
# one loop.
dists_one = classifier.compute_distances_one_loop(X_test)
# compute the differeces between the two methods
differeces = np.linalg.norm(dists - dists_one, ord= 'fro')
if differeces<0.001:
    print('Good, the two method give the same results.')
else:
    print('The distance is different')

# Now we use the method without any loop
dists_non = classifier.compute_distances_no_loops(X_test)
# compute the differeces between the two methods
differeces = np.linalg.norm(dists - dists_non, ord= 'fro')
if differeces<0.001:
    print('Good, The differece is %f'% differeces)
else:
    print('The distance is different')
    
#Let's compute how fast the implementations are

def time_function(f, *args):
    '''
    Call a function f with args and return  the time (in seconds)
    that it took to execute
    '''
    import time 
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic 

two_loops_time = time_function(classifier.compute_distances_two_loops, X_test)
print('Two loops took about %f seconds' % two_loops_time)

one_loops_time = time_function(classifier.compute_distances_one_loop, X_test)
print('One loop took about %f seconds' % one_loops_time)

non_loops_time = time_function(classifier.compute_distances_no_loops,X_test)
print('No loop took about %f seconds' % non_loops_time)
# The vectorlize method with no loop is much faster than the normal methods


#Cross validation
num_folds = 5
k_choices = [1,3,5,8,10,12,15,20,50,100]

X_train_folds = []
Y_train_folds = []

#split up the training data into folds.  
# using function numpy array_split

X_train_folds = np.array(np.array_split(X_train, num_folds))
Y_train_folds = np.array(np.array_split(y_train, num_folds))

# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary. 
k_to_accuracies = {}   
for k in k_choices:
    for j in range(num_folds):
        p=[x for x in range(num_folds) if x!=j]
        num_test=Y_train_folds[j][0]
        X_train= np.concatenate(X_train_folds[p])
        Y_train = np.concatenate(Y_train_folds[p])
        classifier_k = KNearestNeighbor()
        classifier_k.train(X_train,Y_train)
        dist=classifier_k.compute_distances_no_loops(X_train_folds[j])
        y_predict=classifier_k.predict_labels(dist,k)
        num_correct= np.sum(y_predict==Y_train_folds[j])
        accuracy=float(num_correct/num_test)
        k_to_accuracies.setdefault(k,[]).append(accuracy)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))
        
#plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()       
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)              # Calculate the scores of all classes
    correct_class_score = scores[y[i]] 
    for j in xrange(num_classes):
      if j == y[i]:         #if we choose the right class no loss
        continue
      margin = scores[j] - correct_class_score + 1 # else we calculate loss 
      if margin > 0:                               #note delta = 1                                                     
        loss += margin
        #calculate the dW, 
        dW[:,j] += X[i,:].T
        dW[:,y[i]] -= X[i,:].T
        
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  delta = 1
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  num_train = X.shape[0]
  #find the right scores
  scores_right = scores[np.arange(num_train),y]
  #reshape it to (n,1)
  scores_right = np.reshape(scores_right, (num_train, -1))
  #calculate the margin
  L = scores - scores_right + delta
  L = np.maximum(0,L)
  # set the margin of right scores to 0
  L[np.arange(num_train),y] = 0
  # calculate the loss and add regulation
  loss = np.sum(L)/num_train
  loss+= 0.5*reg*np.sum(W*W)
  # calculate  dW (j=yi -xi  j!=yi xi)
  # set a mask for all the margin>0
  L[L>0] =1.0
  # for a xi sum all L>0 and set it to L[yi]
  j_sum = np.sum(L,axis =1)
  L[np.arange(num_train),y] = -j_sum
  #calculate dW with only one vectorized line
  dW += np.dot(X.T,L)/num_train +reg*W

  

  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

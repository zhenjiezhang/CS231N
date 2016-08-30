import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_classes=y.max()+1
  num_samples=X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  out=np.zeros((num_samples,num_classes))
  for s in xrange(num_samples):
    for c in xrange(num_classes):
        out[s][c]=np.exp(X[s].dot(W[:,c]))        
  out/=out.sum(axis=1)[:,np.newaxis]

  for i in xrange(num_samples):
    loss-=np.log(out[i][y[i]])
    for j in xrange(num_classes):
        dW[:,j]+=X[i]*out[i][j]
    dW[:,y[i]]-=X[i]
  dW/=num_samples
  dW+=reg*W
  
  loss/=num_samples
  loss+=0.5*reg*np.sum(W*W)

  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes=y.max()+1
  num_samples=X.shape[0]
    
  out=np.exp(X.dot(W))
  out/=out.sum(axis=1)[:,np.newaxis]
  loss-=sum([np.log(out[i,y[i]]) for i in xrange(num_samples)])/num_samples
  loss+=0.5*reg*np.sum(W*W)

  out=[[out[i][j]-(0 if y[i]!=j else 1) for j in xrange(num_classes)] for i in xrange(num_samples)]
  dW=X.T.dot(out)/num_samples
  dW+=reg*W  
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


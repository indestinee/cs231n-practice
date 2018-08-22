import numpy as np
from random import shuffle

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
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  for i in range(num_train):
      dl = 1.0 / num_train
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]

      scores = (scores - correct_class_score + 1 > 0) * 1 
      scores[y[i]] = 0

      ds = scores * dl
      ds[y[i]] -= scores.sum() * dl
      for k in range(W.shape[1]):
          dW[:, k] += ds[k] * X[i, :]
  dW += 2 * W * reg
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_class_score = scores[range(X.shape[0]), y]
  dist = np.maximum(0, scores - correct_class_score.reshape(-1, 1) + 1)
  dist[range(X.shape[0]), y] = 0
  loss = dist.sum() / X.shape[0]
  loss += reg * (W**2).sum()
  pass
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
  dist = (dist > 0) * 1 / X.shape[0]
  dist[range(X.shape[0]), y] -= dist.sum(axis=1)
  '''
    dL/ds = dist
    s[i, j] = sum{X[i, k] * W[k, j]}
    ds[i, j]/dW[k, j] = sum(X[i, k])
    dL/dW[k, j] = sum{X[i, k] * dist[i, j]}
    dL/dW[k, j] = sum{X-1[k, i] * dist[i, j]}
    dL/dW = X-1.dot(dist)
  '''
  dW = X.transpose().dot(dist)
  dW += 2 * reg * W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

from builtins import range
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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    #그러니까 우선 num_classes 에 class들을 저장하고
    #num_train 에 학습할 batch들을 저장하고
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    #loss로 초기화 한 뒤에
    for i in range(num_train):
        #f = WX 계산하니까 np.dot으로 계산하고
        scores = X[i].dot(W)
        #맞는 class score는 scores[y[i]] 로 구하고
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            #둘이 같으면 넘어가고
            if j == y[i]:
                continue
            #다른데 마진 값이 0보다 클 경우에
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                #loss는 마진 값들의 합이고
                loss += margin
                #간단하게 생각하면 여기 식이 하나 있는데 말이지... max(0,Sj-syi+1) 이라서 이걸 미분하면
                #y[i] 에 미분값을 빼주고 j의 미분값은 더해주는 형식이라는 것
                #근데 input픽셀 값을 전부 더해주는게 사실상 미분값과 다를게 없다! 와우!
                #https://mainpower4309.tistory.com/28?category=867405
                dw[:,y[i]] = dw[:,y[i]] - X[i]
                dw[:,j] = dw[:,j] + X[i]
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dw /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dw += 2*reg*W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.matmul(X,W)
    correct_class_score = scores[range(X.shape[0]),y]
    delta = np.ones(scores.shape)
    delta[range(X.shape[0]),y] = 0
    margin = np.maximum(0,scores-np.reshape(correct_class_score,(correct_class_score.shape[0],1))+delta)
    loss += np.mean(np.sum(margin,axis = 1))
    loss += reg *np.sum(W*W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    margin_except_itself = np.zeros(margin.shape)
    margin_except_itself[margin>0] = 1
    margin_except_itself[range(X.shape[0]),y] = -np.sum(margin_except_itself,axis = 1)
    
    dW = np.matmul(X.T,margin_except_itself) / X.shape[0]

    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

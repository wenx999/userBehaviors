

from math import log, exp
from random import random
from collections import defaultdict


def logSumExp(logs):
    '''
    Arguments:
        logs - a list of log(probabilities)
    Returns: 
        log(sum(exp(logs)))
    '''
    a= min(logs)
    return a+log(sum(exp(i-a) for i in logs))

class likelihoods:
    '''
    A class of likelihood functions supported by the HMM class.
    Types supported: 
        poisson
        multinomial
        binomial
        gaussian
        linear regression
        logistic regression
    '''
    def __init__(self, type):



class HMM:
    '''
    A class for defining arbitrary hidden markov models
    '''

    def __init__(self, n_classes, data_types):
        '''
        Arguments:
            n_classes - int number of classes to fit
            data_types - distributions to fit, same length as data
        '''
        self.n_classes= n_classes
        self.data_types= data_types
        init_probs= [random() for i in range(n_classes)]
        self.init_probs= [i/sum(init_probs) for i in init_probs]
        trans_probs= defaultdict(dict)
        for i in range(n_classes):
            for j in range(n_classes):
                trans_probs[i][j]= random()
            total= sum(trans_probs[i].values())
            for j in range(n_classes):
                trans_probs[i][j]= trans_probs[i][j]/total
        self.trans_probs= trans_probs


    def forwardPass(self):
        pass

    def backwardPass(self):
        pass

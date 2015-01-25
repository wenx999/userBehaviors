

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


class HMM:
    '''
    Just a multinomial HMM first
    '''
    def __init__(self, n_states, n_words, init_probs= False, trans_probs= False, emit_probs= False):
        '''
        Arguments:
            n_states - int number of classes to fit
            n_words - int number of emissions
        '''
        self.n_states= n_states
        init_probs= [random() for i in range(n_states)]
        init_probs= [i/sum(init_probs) for i in init_probs]
        exit_probs= [random() for i in range(n_states)]
        exit_probs= [i/sum(init_probs) for i in init_probs]
        trans_probs= defaultdict(dict)
        for i in range(n_states):
            for j in range(n_states):
                trans_probs[i][j]= random()
            total= sum(trans_probs[i].values())
            for j in range(n_states):
                trans_probs[i][j]= trans_probs[i][j]/total
        emit_probs= defaultdict(dict)
        for i in range(n_states):
            for j in range(n_words):
                emit_probs[i][j]= random()
            total= sum(emit_probs[i].values())
            for w in range(n_words):
                emit_probs[i][w]= emit_probs[i][w]/total
        self.trans_probs= trans_probs
        self.init_probs= init_probs
        self.exit_probs= exit_probs
        self.emit_probs= emit_probs


    def forwardPass(self, sequence):
        stateLikelihoodSeq= []
        for ind, obs in enumerate(sequence):
            if ind==0:
                stateLikelihoodSeq.append([log(self.init_probs[i])+log(self.emit_probs[i][obs]) for i in range(self.n_states)])
            else:
                stateLikelihoods= []
                for curr in range(self.n_states):
                    logs= []
                    for prev in range(self.n_states):
                        logs.append(stateLikelihoodSeq[ind-1][prev]+log(self.trans_probs[prev][curr])+log(self.emit_probs[curr][obs]))
                    stateLikelihoods.append(logSumExp(logs))
        return stateLikelihoodSeq


    def backwardPass(self, sequence):
        stateLikelihoodSeq= []
        for ind, obs in reversed(list(enumerate(sequence))):
            if ind==0:
                stateLikelihoodSeq.append([log(self.exit_probs[i])+log(self.emit_probs[i][obs]) for i in range(self.n_states)])
            else:
                stateLikelihoods= []
                for prev in range(self.n_states):
                    logs= []
                    for curr in range(self.n_states):
                        logs.append(stateLikelihoodSeq[ind-1][prev]+log(self.trans_probs[prev][curr])+log(self.emit_probs[curr][obs]))
                    stateLikelihoods.append(logSumExp(logs))
        return stateLikelihoodSeq              






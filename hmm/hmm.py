

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
    a= max(logs)
    return a+log(sum(exp(i-a) for i in logs))


class HMM:
    '''
    Just a multinomial HMM first
    '''
    def __init__(self, n_states, n_words, sequence= None, init_probs= None, exit_probs=None, trans_probs= None, emit_probs= None):
        '''
        Arguments:
            n_states - int number of classes to fit
            n_words - int number of emissions
        '''
        if not init_probs:
            init_probs= [random() for i in range(n_states)]
            init_probs= [i/sum(init_probs) for i in init_probs]
        if not exit_probs:
            exit_probs= [random() for i in range(n_states)]
            exit_probs= [i/sum(init_probs) for i in init_probs]
        if not trans_probs:
            trans_probs= defaultdict(dict)
            for i in range(n_states):
                for j in range(n_states):
                    trans_probs[i][j]= random()
                total= sum(trans_probs[i].values())
                for j in range(n_states):
                    trans_probs[i][j]= trans_probs[i][j]/total
        if not emit_probs:
            emit_probs= defaultdict(dict)
            for i in range(n_states):
                for j in range(n_words):
                    emit_probs[i][j]= random()
                total= sum(emit_probs[i].values())
                for w in range(n_words):
                    emit_probs[i][w]= emit_probs[i][w]/total
        
        self.n_states= n_states
        self.trans_probs= trans_probs
        self.init_probs= init_probs
        self.exit_probs= exit_probs
        self.emit_probs= emit_probs
        self.sequence= sequence


    def forwardPass(self):
        stateLikelihoodSeq= []
        for ind, obs in enumerate(self.sequence):
            if ind==0:
                stateLikelihoodSeq.append([log(self.init_probs[i])+log(self.emit_probs[i][obs]) for i in range(self.n_states)])
            else:
                stateLikelihoods= []
                for curr in range(self.n_states):
                    logs= []
                    for prev in range(self.n_states):
                        l_prevState= stateLikelihoodSeq[ind-1][prev] 
                        l_transPrevCurr= log(self.trans_probs[prev][curr])
                        l_emitCurrState= log(self.emit_probs[curr][obs])
                        logs.append(l_prevState+l_transPrevCurr+l_emitCurrState)
                    stateLikelihoods.append(logSumExp(logs))
                stateLikelihoodSeq.append(stateLikelihoods)
        return stateLikelihoodSeq


    def backwardPass(self):
        stateLikelihoodSeq= []
        for ind, obs in reversed(list(enumerate(self.sequence))):
            if ind==len(self.sequence)-1:
                stateLikelihoodSeq.append([log(self.exit_probs[i])+log(self.emit_probs[i][obs]) for i in range(self.n_states)])
            else:
                stateLikelihoods= []
                for prev in range(self.n_states):
                    logs= []
                    for curr in range(self.n_states):
                        l_currState= stateLikelihoodSeq[len(self.sequence)-ind-2][curr]
                        l_transPrevCurr= log(self.trans_probs[prev][curr])
                        l_emitCurrState= log(self.emit_probs[curr][obs])
                        logs.append(l_currState+l_transPrevCurr+l_emitCurrState)
                    stateLikelihoods.append(logSumExp(logs))
                stateLikelihoodSeq.append(stateLikelihoods)
        return stateLikelihoodSeq


    def updateSequence(self, sequence):
        self.sequence= sequence


    def updateParams(self, sequence):
        self.updateSequence(sequence)
        forwardProbs= forwardPass()
        backwardProbs= backwardPass()
        






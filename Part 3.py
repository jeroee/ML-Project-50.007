import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from Part 2 import *

EN_train = 'EN/train'
SG_train = 'SG/train'
CN_train = 'CN/train'
EN_test = 'EN/dev.in'
SG_test = 'SG/dev.in'
CN_test = 'CN/dev.in'

# load training file


def load_train_trans(training_file):
    f = open(training_file)
    ls_state = ['START']
    for line in f:
        item = line.strip('\n').split(' ')
        if len(item) == 2:
            ls_state.append(item[1])
        elif len(item) < 2:
            ls_state.append('STOP')
            ls_state.append('START')
    ls_state.pop(-1)
    return ls_state

# load test files

def load_test(testing_file):
    ls = []
    f = open(testing_file)
    for line in f:
        ls.append(line.strip('\n'))
    df = pd.DataFrame(ls)
    df.columns = ['word']
    return df

# Part 1 - Transition Parameter
def transition_matrix(training_list):
    count = Counter(training_list)
    list_key = list(count.keys())
    rls_matrix = pd.DataFrame(columns=list_key, index=list_key)
    for (x, y), c in Counter(zip(temp, temp[1:])).items():
        rls_matrix.loc[[x], [y]] = c/count[x]
    rls_matrix = rls_matrix.fillna(value=0)
    rls_matrix = rls_matrix.drop(columns='START')
    rls_matrix = rls_matrix.drop(index='STOP')
    return rls_matrix


# Implement the Function
temp = load_train_trans(EN_train)
tran_matrix = transition_matrix(temp)
print(tran_matrix)

# Part 2 Vertibri Algo
m = open('EN/dev.in', encoding="utf8")
ls=[]
big_ls=[]
for line in m:
    item=line.strip('\n')
    if item=='':
        big_ls.append(ls)
        ls=[]
    elif item!='':
        ls.append(item)
# PseudoCode
# big_v=[]
# for i in big_ls:  #for each sentene
#     forward_steps = len(i)+1 
#     for j in forward_steps: #for each layer
#         if j==0:  #start to first layer
#             small_v= [a*b for a,b in zip([start->all states][:-1],[all states --> i[j] #word in sentence])]
#             big_v.append(small_v)
#         elif j!=0 & j!=forward_steps: #not first or last step
#             small_v=[a*b*c for a,b,c in zip([small_v],[state-> state],[ state -> word])]
#             big_v.append(small_v)
#         else: #if last step

def viterbi(y, A, B):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    ----------
    y : array (T,)
        Observation state sequence. int dtype.
    A : array (K, K)
        State transition matrix. See HiddenMarkovModel.state_transition  for
        details.
    B : array (K, M)
        Emission matrix. See HiddenMarkovModel.emission for details.
    Pi: optional, (K,)
        Initial state probabilities: Pi[i] is the probability x[0] == i. If
        None, uniform initial distribution is assumed (Pi[:] == 1/K).

    Returns
    -------
    x : array (T,)
        Maximum a posteriori probability estimate of hidden state trajectory,
        conditioned on observation sequence y under the model parameters A, B,
        Pi.
    T1: array (K, T)
        the probability of the most likely path so far
    T2: array (K, T)
        the x_j-1 of the most likely path so far
    """
    # Cardinality of the state space
    K = A.shape[0]
    # Initialize the priors with default (uniform dist) if not given by caller
    Pi = np.full(K, 1 / K)
    T = len(y)
    T1 = np.empty((K, T), 'd')
    T2 = np.empty((K, T), 'B')

    # Initilaize the tracking tables from first observation
    T1[:, 0] = Pi * B[:, y[0]]
    T2[:, 0] = 0

    # Iterate throught the observations updating the tracking tables
    for i in range(1, T):
        T1[:, i] = np.max(T1[:, i - 1] * A.T * B[np.newaxis, :, y[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * A.T, 1)

    # Build the output, optimal model trajectory
    x = np.empty(T, 'B')
    x[-1] = np.argmax(T1[:, T - 1])
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]

    return x, T1, T2


viterbi(temp, tran, )
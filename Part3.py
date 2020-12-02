import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

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

# Part 1
def relation_matrix(training_list):
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
rls_matrix = relation_matrix(temp)
print(rls_matrix)

'''
VERTIBRI ALGORITHM TBC
'''
def log(x, inf_replace=-1000):
    out = np.log(x)
    out[~np.isfinite(out)] = inf_replace
    return out
logged_emission = log(emission_matrix)
logged_transition = log(transition_matrix)
transition_np = logged_transition.drop(['START']).drop('STOP',axis=1).to_numpy()

# test for one document
tags = argmax(emission_matrix)   # vocab of words
Vertibri = []
document = big_ls[1]
# print(document)
forward_steps = len(document)+1
for i in range(forward_steps):
    if i == 0: # for from START to first layer
        if document[i] in tags.keys():
            layer = [t+e for t,e in zip(logged_transition.loc['START'].drop('STOP'), logged_emission[document[i]])]
        elif document[i] not in tags.keys():
            layer = [t+e for t,e in zip(logged_transition.loc['START'].drop('STOP'), logged_emission['#UNK#'])]
        Vertibri.append(layer)
        print(type(Vertibri[-1]))
    elif i!=0 and i!=forward_steps-1: #not first or last step
        prev_layer_prob = Vertibri[-1]*21
        prev_layer_prob = np.array(prev_layer_prob).reshape(21,21).T
        m = prev_layer_prob + transition_np
        if document[i] in tags.keys():
            emission_ls = logged_emission[document[i]].tolist()*21
            emission_np = np.array(emission_ls).reshape(21,21)
        elif document[i] not in tags.keys():
            emission_ls = logged_emission['#UNK#'].tolist()*21
            emission_np = np.array(emission_ls).reshape(21,21)
        matrix = (m + emission_np)
        layer = np.amax(matrix,0)
        Vertibri.append(layer.tolist())
    elif i == forward_steps-1:
        prev_layer_prob = np.array(Vertibri[-1])
        last = logged_transition.drop('START')['STOP'].tolist()
        layer = prev_layer_prob+last
        Vertibri.append(layer.tolist())

state_order = []
states = emission_matrix.index.tolist()
# Vertibri.pop(0)
for layer in Vertibri:
    position = layer.index(max(layer))
    state_order.append(states[position])


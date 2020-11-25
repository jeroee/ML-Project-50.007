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
def load_train(training_file):
    f = open('EN/train')
    ls_state = ['START']
    for line in f:
        item = line.strip('\n').split(' ')
        if len(item)==2:
            ls_state.append(item[1])
        elif len(item)<2:
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
    df.columns=['word']
    return df

def relation_matrix(training_list):
    count = Counter(temp)
    list_key = list(count.keys())
    rls_matrix = pd.DataFrame(columns = list_key, index = list_key)
    for (x,y), c in Counter(zip(temp, temp[1:])).items():
        rls_matrix.loc[[x], [y]] = c/count[x]
    rls_matrix = rls_matrix.fillna(value=0)
    rls_matrix = rls_matrix.drop(columns = 'START')
    rls_matrix = rls_matrix.drop(index = 'STOP')
    return rls_matrix


#Implement the Function
temp = load_train(EN_train)
rls_matrix = relation_matrix(temp)

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from Part_2 import *

# EN_train = 'EN/train'
# SG_train = 'SG/train'
# CN_train = 'CN/train'
# EN_test = 'EN/dev.in'
# SG_test = 'SG/dev.in'
# CN_test = 'CN/dev.in'

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
    for (x, y), c in Counter(zip(training_list, training_list[1:])).items():
        rls_matrix.loc[[x], [y]] = c/count[x]
    rls_matrix = rls_matrix.fillna(value=0)
    rls_matrix = rls_matrix.drop(columns='START')
    rls_matrix = rls_matrix.drop(index='STOP')
    return rls_matrix


# Implement the Function
# temp = load_train_trans(EN_train)
# tran_matrix = transition_matrix(temp)
# print(tran_matrix)

# Part 2 Vertibri Algo
def load_vertibri_test(file):
    m = open(file, encoding="utf8")
    ls = []
    big_ls = []
    for line in m:
        item = line.strip('\n')
        if item == '':
            big_ls.append(ls)
            ls = []
        elif item != '':
            ls.append(item)
    return big_ls


'''
Full Vertibri Algorithm
'''


def log(x, inf_replace=-1000):
    x = x.replace(0, np.nan)
    out = np.log(x)
    out = out.replace(np.nan, -1000)
    # out[~np.isfinite(out)] = inf_replace
    return out


def Vertibri(documents, emission_matrix, transition_matrix):
    # forward propogation
    logged_emission = log(emission_matrix)
    logged_transition = log(transition_matrix)
    transition_np = logged_transition.drop(
        ['START']).drop('STOP', axis=1).to_numpy()
    states = emission_matrix.index.tolist()
    big_ls = []
    Vertibri = []
    print('forward propogation')
    for document in tqdm(documents):
        forward_steps = len(document)+1
        for i in range(forward_steps):
            if i == 0:  # for from START to first layer
                if document[i] in tags.keys():
                    layer = [
                        t+e for t, e in zip(logged_transition.loc['START'].drop('STOP'), logged_emission[document[i]])]
                elif document[i] not in tags.keys():
                    layer = [
                        t+e for t, e in zip(logged_transition.loc['START'].drop('STOP'), logged_emission['#UNK#'])]
                Vertibri.append(layer)
            elif i != 0 and i != forward_steps-1:  # not first or last step
                prev_layer_prob = Vertibri[-1]*21
                prev_layer_prob = np.array(prev_layer_prob).reshape(21, 21).T
                m = prev_layer_prob + transition_np
                if document[i] in tags.keys():
                    emission_ls = logged_emission[document[i]].tolist()*21
                    emission_np = np.array(emission_ls).reshape(21, 21)
                elif document[i] not in tags.keys():
                    emission_ls = logged_emission['#UNK#'].tolist()*21
                    emission_np = np.array(emission_ls).reshape(21, 21)
                matrix = (m + emission_np)
                layer = np.amax(matrix, 0)
                Vertibri.append(layer.tolist())
            elif i == forward_steps-1:
                prev_layer_prob = np.array(Vertibri[-1])
                last = logged_transition.drop('START')['STOP'].tolist()
                layer = prev_layer_prob+last
                # gets the position of the maximum in the last one
                pos_max_layer = (layer.tolist().index(max(layer)))
                Vertibri.append(layer.tolist())
        big_ls.append(Vertibri)
    # -------------------------------------------------------------------------------
    # backward propogation

    # Stores the states
    big_state = []
    state_order = []
    print('backward propogation')
    for Vertibri in tqdm(big_ls):
        # Create a trimmed one with all less the stop column
        Vertibri_trim = Vertibri[:-1]

        # Iterate through each layer find the argmax for the layer and continue
        for layer in Vertibri_trim[::-1]:
            # List that holds the summed value of the last term with the second last layer
            intermediate_ls = []
            intermediate_ls = [x+transition_np[layer.index(x), pos_max_layer] for x in layer]
            # Find the argmax for the layer
            pos_max_value = np.max(intermediate_ls)
            pos_max_layer = np.argmax(intermediate_ls)
            # Insert into the state order
            state_order.insert(0, states[pos_max_layer])
            # Update the value of position_last_max
            position_last_max = pos_max_value
        big_state.append(state_order)
    # -------------------------------------------------------------------------------
'''
END
'''
# tagging words


def tag_system_3(states, df):
    states_ls = []
    try: 
        for state in states:
            for i in state:
                states_ls.append(i)
            states_ls.append(' ')
        df['states'] = states_ls
    except: 
        pass
    return df


# file paths
EN_train = 'EN/train'
SG_train = 'SG/train'
CN_train = 'CN/train'
EN_test = 'EN/dev.in'
SG_test = 'SG/dev.in'
CN_test = 'CN/dev.in'
EN_pred_3 = 'EN/dev_p3.pred'
SG_pred_3 = 'SG/dev_p3.pred'
CN_pred_3 = 'CN/dev_p3.pred'
files = [['EN', EN_train, EN_test, EN_pred_3],
         ['SG', SG_train, SG_test, SG_pred_3],
         ['CN', CN_train, CN_test, CN_pred_3]]

print('Starting Part 3')
start_time = time.time()
for i in files:
    print(f'starting {i[0]}')
    print('getting emission matrix')
    df_train = load_train(i[1])
    # df_test = load_test(i[2])
    emission_matrix = createMatrix(df_train)
    emission_matrix = emissionMatrix_special(df_train, emission_matrix)
    tags = argmax(emission_matrix)
    print('getting transition matrix')

    ls = load_train_trans(i[1])
    transition_matrix = transition_matrix(ls)
    print('performing Vertibri Algo')
    documents = load_vertibri_test(i[1])
    states = Vertibri(documents, emission_matrix, transition_matrix)
    print('tagging states to documents')
    df_test = load_test(i[2])
    df_test = tag_system_3(states, df_test)
    save_df(df_test, i[3])

    # df_output = tag_system(tags, df_test)
    # save_df(df_output, i[3])
    print(f'{i[0]} completed')

print('Part 3 Complete')
print(f'time elapsed {time.time()-start_time} seconds')


# def log(x, inf_replace=-1000):
#     out = np.log(x)
#     out[~np.isfinite(out)] = inf_replace
#     return out

# logged_emission = log(emission_matrix)
# logged_transition = log(transition_matrix)
# transition_np = logged_transition.drop(['START']).drop('STOP',axis=1).to_numpy()

# # test for one document
# states = emission_matrix.index.tolist() #get the states out in a
# tags = argmax(emission_matrix)   # vocab of words
# Vertibri = []
# document = big_ls[1]
# # print(document)
# forward_steps = len(document)+1
# for i in range(forward_steps):
#     if i == 0: # for from START to first layer
#         if document[i] in tags.keys():
#             layer = [t+e for t,e in zip(logged_transition.loc['START'].drop('STOP'), logged_emission[document[i]])]
#         elif document[i] not in tags.keys():
#             layer = [t+e for t,e in zip(logged_transition.loc['START'].drop('STOP'), logged_emission['#UNK#'])]
#         Vertibri.append(layer)
#     elif i!=0 and i != forward_steps-1: #not first or last step
#         prev_layer_prob = Vertibri[-1]*21
#         prev_layer_prob = np.array(prev_layer_prob).reshape(21,21).T
#         m = prev_layer_prob + transition_np
#         if document[i] in tags.keys():
#             emission_ls = logged_emission[document[i]].tolist()*21
#             emission_np = np.array(emission_ls).reshape(21,21)
#         elif document[i] not in tags.keys():
#             emission_ls = logged_emission['#UNK#'].tolist()*21
#             emission_np = np.array(emission_ls).reshape(21,21)
#         matrix = (m + emission_np)
#         layer = np.amax(matrix,0)
#         Vertibri.append(layer.tolist())
#     elif i == forward_steps-1:
#         prev_layer_prob = np.array(Vertibri[-1])
#         last = logged_transition.drop('START')['STOP'].tolist()
#         layer = prev_layer_prob+last
#         #gets the position of the maximum in the last one
#         position_last_max = (layer.tolist().index(max(layer)))
#         print(states[position_last_max])
#         Vertibri.append(layer.tolist())


# #Stores the states
# state_order = []
# #Create a trimmed one with all less the stop column
# Vertibri_trim = Vertibri[:-1]

# #Iterate through each layer find the argmax for the layer and continue
# for layer in Vertibri_trim[::-1]:
#     #List that holds the summed value of the last term with the second last layer
#     intermediate_ls = []
#     intermediate_ls = [x+position_last_max for x in layer]
#     #Find the argmax for the layer
#     pos_max_value = np.max(intermediate_ls)
#     pos_max_layer = np.argmax(intermediate_ls)
#     #Insert into the state order
#     state_order.insert(0,states[pos_max_layer])
#     #Update the value of position_last_max
#     position_last_max = pos_max_value

# print(state_order)

# for layer in range(len(Vertibri)-1):
#     last_index = len(Vertibri)-2
#     second_last_index = last_index-1
#     print(np.argmax(Vertibri[last_index]))

# print(second_last_matrix)


# state_order = []
# states = emission_matrix.index.tolist()
# for layer in Vertibri:
#     position = layer.index[max(layer)]
#     # states = emission_matrix.index.tolist()
#     state_order.append(states[position])
# print(state_order)

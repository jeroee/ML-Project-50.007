import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from Part_2 import *

# load train file for transition matrix


def load_train_trans(training_file):
    f = open(training_file, encoding="utf8")
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


# load test file dev.in into a single column df


def load_test(testing_file):
    ls = []
    f = open(testing_file, encoding="utf8")
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


# Part 2 Vertibri Algo
# load test file into a nested list


def pre_vertibri_load(file):
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


def log(x, inf_replace=-1000):
    x = x.replace(0, np.nan)
    out = np.log(x)
    out = out.replace(np.nan, -1000)
    # out[~np.isfinite(out)] = inf_replace
    return out


def getHelpers(emission_matrix, transition_matrix):
    logged_emission = log(emission_matrix)
    logged_transition = log(transition_matrix)
    transition_np = logged_transition.drop(
        ['START']).drop('STOP', axis=1).to_numpy()
    states = emission_matrix.index.tolist()
    return logged_emission, logged_transition, transition_np, states


def single_Viterbi(tweet, logged_emission, logged_transition, transition_np, states, tags):
    len_states = len(states)
    Viterbi = []
    forward_steps = len(tweet)+1
    for i in range(forward_steps):
        if i == 0:  # for from START to first layer
            if tweet[i] in tags.keys():
                layer = [
                    t+e for t, e in zip(logged_transition.loc['START'].drop('STOP'), logged_emission[tweet[i]])]
            elif tweet[i] not in tags.keys():
                layer = [
                    t+e for t, e in zip(logged_transition.loc['START'].drop('STOP'), logged_emission['#UNK#'])]
            Viterbi.append(layer)  # append first layer
        elif i == 1:  # if second layer
            prev_layer_prob = Viterbi[-1]*len_states
            prev_layer_prob = np.array(prev_layer_prob).reshape(
                len_states, len_states).T
            m = prev_layer_prob + transition_np
            if tweet[i] in tags.keys():
                emission_ls = logged_emission[tweet[i]].tolist()*len_states
                emission_np = np.array(emission_ls).reshape(
                    len_states, len_states)
            elif tweet[i] not in tags.keys():
                emission_ls = logged_emission['#UNK#'].tolist()*len_states
                emission_np = np.array(emission_ls).reshape(
                    len_states, len_states)
            matrix = (m + emission_np)
            layer = np.sort(np.partition(matrix.T, -3, axis=1)[:, -3:], axis=1)
            # layer = np.amax(matrix, axis=0)
            Viterbi.append(layer.tolist())
        elif i != 0 and i != forward_steps-1:  # not first,second or last layer
            # prev_layer_prob = Viterbi[-1]*len_states
            # prev_layer_prob = np.array(prev_layer_prob).reshape(
            #     len_states, len_states).T
            # m = prev_layer_prob + transition_np
            def f(x): return np.repeat(x, len_states)
            prev_layer_prob = np.apply_along_axis(f, 1, np.array(Viterbi[-1]))
            m = prev_layer_prob + \
                np.hstack((transition_np, transition_np, transition_np))
            if tweet[i] in tags.keys():
                emission_ls = logged_emission[tweet[i]].tolist()*len_states
                emission_np = np.array(emission_ls).reshape(
                    len_states, len_states)
            elif tweet[i] not in tags.keys():
                emission_ls = logged_emission['#UNK#'].tolist()*len_states
                emission_np = np.array(emission_ls).reshape(
                    len_states, len_states)
            matrix = (m + np.hstack((emission_np, emission_np, emission_np)))
            layer = np.sort(np.partition(matrix, -3, axis=1)[:, -3:], axis=1)
            Viterbi.append(layer.tolist())
        elif i == forward_steps-1:
            prev_layer_prob = np.array(Viterbi[-1])
            last = np.array(logged_transition.drop('START')
                            ['STOP']).reshape(len_states, 1)
            # last = logged_transition.drop('START')['STOP'].tolist()
            layer = prev_layer_prob+np.hstack((last, last, last))
            layer_flatten = layer.flatten()
            value = np.partition(layer_flatten, -3)[-3]
            pos_max_layer = layer_flatten.tolist().index(value)//3
            # gets the position of the maximum in the last one
            # pos_max_layer = (layer.tolist().index(max(layer)))
            Viterbi.append(layer.tolist())
            print(Viterbi)
            print(len(Viterbi))
            print(pos_max_layer)
            break
            # this step is correct alr, forward prop done
    # back prop starts here
    Viterbi_trim = Viterbi[:-1]
    state_order = []
    # Iterate through each layer find the argmax for the layer and continue
    for layer in Viterbi_trim[::-1]:
        # List that holds the summed value of the last term with the second last layer
        intermediate_arr = np.array(layer)
        intermediate_ls = intermediate_arr + transition_np[:, pos_max_layer]
        # Find the argmax for the layer
        pos_max_value = np.max(intermediate_ls)
        pos_max_layer = np.argmax(intermediate_ls)
        # Insert into the state order
        state_order.insert(0, states[pos_max_layer])
    return state_order


file_train = 'EN/train'
file_test = 'EN/dev.in'
df_train = load_train(file_train)
emission_matrix = createMatrix(df_train)
emission_matrix = emissionMatrix_special(df_train, emission_matrix)
tags = argmax(emission_matrix)
ls = load_train_trans(file_train)
transition_matrix1 = transition_matrix(ls)
logged_emission, logged_transition, transition_np, states = getHelpers(
    emission_matrix, transition_matrix1)
tweets = pre_vertibri_load(file_test)
tweet = tweets[3]  # test with second item
single_state = single_Viterbi(
    tweet, logged_emission, logged_transition, transition_np, states, tags)
print(single_state)


# this is cfm alr
# def output(file_train, file_test, path):
#     # file_train = 'EN/train'
#     # file_test = 'EN/dev.in'
#     df_train = load_train(file_train)
#     emission_matrix = createMatrix(df_train)
#     emission_matrix = emissionMatrix_special(df_train, emission_matrix)
#     tags = argmax(emission_matrix)
#     print('getting transition matrix')
#     ls = load_train_trans(file_train)
#     transition_matrix1 = transition_matrix(ls)
#     logged_emission, logged_transition, transition_np, states = getHelpers(
#         emission_matrix, transition_matrix1)
#     tweets = pre_vertibri_load(file_test)
#     # ------------------ big vertibri ---------------------
#     big_states = []
#     for tweet in tqdm(tweets):
#         single_state = single_Viterbi(tweet, logged_emission, logged_transition,
#                                       transition_np, states, tags)
#         single_state.append(' ')
#         big_states.append(single_state)
#     # flat map this shit
#     states = [item for single_state in big_states for item in single_state]
#     # ---------------------
#     df = load_test(file_test)
#     df['states'] = states
#     save_df(df, path)


# print(test)
# print(len(single_state))
# print(single_state)


# # file paths
EN_train = 'EN/train'
SG_train = 'SG/train'
CN_train = 'CN/train'
EN_test = 'EN/dev.in'
SG_test = 'SG/dev.in'
CN_test = 'CN/dev.in'
EN_pred_3 = 'EN/dev_p3.pred'
SG_pred_3 = 'SG/dev_p3.pred'
CN_pred_3 = 'CN/dev_p3.pred'

# print('Starting Part 3')
# start_time = time.time()
# output(EN_train, EN_test, EN_pred_3)
# output(SG_train, SG_test, SG_pred_3)
# output(CN_train, CN_test, CN_pred_3)
# print('Part 3 Complete')
# print(f'time elapsed {time.time()-start_time} seconds')

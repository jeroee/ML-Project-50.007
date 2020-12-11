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
            matrix = matrix.T
            # need to include index
            # computing top 3 paths's indexes
            indexes_3 = np.argsort(matrix, axis=1)[:, -3:][:, ::-1]
            # computing top 3 paths values
            values_3 = np.sort(np.partition(
                matrix, -3, axis=1)[:, -3:], axis=1)[:, ::-1]
            # merging index and value in (index,value) tuple
            layer = np.rec.fromarrays([indexes_3, values_3])
            Viterbi.append(layer.tolist())
            # print(Viterbi)
        elif i != 0 and i != forward_steps-1:  # not first,second or last layer
            def f(x): return np.repeat(x, len_states)
            prev_layer_prob = np.apply_along_axis(f, 1, values_3)   # (21x63)
            m = prev_layer_prob + \
                np.hstack((transition_np, transition_np, transition_np)
                          )   # getting logged transition matrix concatanated across 3x   (21x63)
            if tweet[i] in tags.keys():
                emission_ls = logged_emission[tweet[i]].tolist()*len_states
                emission_np = np.array(emission_ls).reshape(
                    len_states, len_states)
            elif tweet[i] not in tags.keys():
                emission_ls = logged_emission['#UNK#'].tolist()*len_states
                emission_np = np.array(emission_ls).reshape(
                    len_states, len_states)
            # getting logged emission matrix concatanated across 3x  (21x63)
            matrix = (m + np.hstack((emission_np, emission_np, emission_np)))
            # need to transpose internally for 1st,2nd,3rd matrices which are concatenated side by side
            # split matrices into 3 smaller matrices. each will be 1st,2nd,3rd from prev nodes
            newarr = np.array_split(matrix, 3, axis=1)
            # transpose all smaller matrices
            transposed = [i.T for i in newarr]
            matrix_transposed = np.hstack(
                (transposed[0], transposed[1], transposed[2]))  # join back again side by side
            # print(matrix_transposed.shape)
            values_3 = np.sort(np.partition(
                matrix_transposed, -3, axis=1)[:, -3:], axis=1)[:, ::-1]  # computing top 3 paths's values
            indexes_3 = np.argsort(matrix_transposed, axis=1)[
                :, -3:][:, ::-1]  # computing top 3 paths's indexes
            # if its >=21 and <42, take value - 21
            arr = np.where((indexes_3 >= 21) & (
                indexes_3 < 42), indexes_3-21, indexes_3)
            # if its >=42 take value -42
            indexes_3 = np.where(arr >= 42, arr-42, arr)
            # merging index and value in (index,value) tuple
            layer = np.rec.fromarrays([indexes_3, values_3])
            Viterbi.append(layer.tolist())
        elif i == forward_steps-1:
            # prev_layer_prob = np.array(Viterbi[-1])
            prev_layer_prob = values_3
            last = np.array(logged_transition.drop('START')
                            ['STOP']).reshape(len_states, 1)  # getting transition matrix to STOP
            layer = prev_layer_prob+np.hstack((last, last, last))
            layer_flatten = layer.flatten().reshape(1, 63)
            indexes = np.argsort(layer_flatten, axis=1)[
                :, -3:][:, ::-1]   # getting the top 3 indexes
            # locating nodes from top 3 indexes
            indexes = np.where(indexes, indexes//3, indexes)
            values = np.sort(np.partition(
                layer_flatten, -3, axis=1)[:, -3:], axis=1)[:, ::-1]  # computing top 3 paths's values
            layer = np.rec.fromarrays([indexes, values])
            # print(layer)
            latest_tuple = layer[0][2]
            current_index = layer[0][2][0]
            # this step is correct alr, forward prop done
    # back prop starts here
    state_order = []
    state_order.append(states[latest_tuple[0]])
    latest_list = list(latest_tuple)
    latest_list.append(current_index)
    current_word = tweet[len(tweet)-1]
    # The format of the new list (not tuple), is going to be [index, value, current index(the row value of it)]
    # Iterate through each layer find the argmax for the layer and continue
    Viterbi_trim = Viterbi[:-1]
    for layer in Viterbi_trim[::-1]:
        # Get the transition probability to add
        transition_prob = logged_transition.at[states[latest_list[0]],
                                               states[latest_list[2]]]
        # Get the emission probability to add
        # 1. get the state which is the row value for the emission matrix
        current_state = states[latest_list[2]]
        # Use .loc to find the emission
        try:
            emission_prob = logged_emission.loc[current_state, current_word]
        except:
            emission_prob = logged_emission.loc[current_state, '#UNK#']
        # Flatten the latest_list which has the 3 values we want to consider
        try:
            flatten = [item for sublist in layer[latest_list[0]]
                       for item in sublist]
            # Get the values to compare
            only_values = flatten[1::3]
            # substract the transition and emission probabilities from value
            back_value = latest_list[1] - emission_prob - transition_prob
            # Substract the values in the only_values by the back value
            minus_values = [x-back_value for x in only_values]
            # Get the minimum index, which will point towards the new tuple we want
            min_position = minus_values.index(min(minus_values))
            # Get the new tuple
            current_index = latest_list[0]
            latest_tuple = layer[latest_list[0]][min_position]
            latest_list = list(latest_tuple)
            latest_list.append(current_index)
            # Append the states
            state_order.insert(0, states[latest_list[0]])
        except:
            flatten = layer[latest_list[0]]
            state_order.insert(0, states[latest_list[0]])
    return state_order


def output(file_train, file_test, path):
    # file_train = 'EN/train'
    # file_test = 'EN/dev.in'
    df_train = load_train(file_train)
    emission_matrix = createMatrix(df_train)
    emission_matrix = emissionMatrix_special(df_train, emission_matrix)
    tags = argmax(emission_matrix)
    print('getting transition matrix')
    ls = load_train_trans(file_train)
    transition_matrix1 = transition_matrix(ls)
    logged_emission, logged_transition, transition_np, states = getHelpers(
        emission_matrix, transition_matrix1)
    tweets = pre_vertibri_load(file_test)
    # ------------------ big vertibri ---------------------
    big_states = []
    for tweet in tqdm(tweets):
        single_state = single_Viterbi(tweet, logged_emission, logged_transition,
                                      transition_np, states, tags)
        single_state.append(' ')
        big_states.append(single_state)
    # flat map this shit
    states = [item for single_state in big_states for item in single_state]
    # ---------------------
    df = load_test(file_test)
    df['states'] = states
    save_df(df, path)


# # file paths
EN_train = 'EN/train'
SG_train = 'SG/train'
CN_train = 'CN/train'
EN_test = 'EN/dev.in'
SG_test = 'SG/dev.in'
CN_test = 'CN/dev.in'
EN_pred_4 = 'EN/dev_p4.pred'
SG_pred_4 = 'SG/dev_p4.pred'
CN_pred_4 = 'CN/dev_p4.pred'

print('Starting Part 4')
start_time = time.time()
output(EN_train, EN_test, EN_pred_4)
print('Part 4 Complete')
print(f'time elapsed {time.time()-start_time} seconds')

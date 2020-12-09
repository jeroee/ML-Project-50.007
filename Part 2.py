import numpy as np
import pandas as pd
from tqdm import tqdm
import time


# load training file
def load_train(training_file):
    df = pd.read_csv(training_file, sep=' ',
                     header=None, error_bad_lines=False)
    df.columns = ['word', 'state']
    return df

# load testing file


def load_test(testing_file):
    ls = []
    f = open(testing_file, encoding="utf8")
    for line in f:
        ls.append(line.strip('\n'))
    df = pd.DataFrame(ls)
    df.columns = ['word']
    return df

# load empty emission probability table


def createMatrix(df):
    # start = time.time()
    columns = df.word.unique().tolist()
    index = df.state.unique().tolist()
    new_df = pd.DataFrame(columns=columns, index=index)
    # print(f'time elapsed {time.time()-start} seconds')
    return new_df

# obtaining emission probabilities from training data


def emissionMatrix(df, emission_matrix):
    '''
    emissionMatrix returns a matrix with the emission parameters matrix.
    df: the dataframe that contains the individual word instance and its label
    emission_matrix: The matrix that has been generated from createMatrix() that will be used to place
        the calculations
    '''
    # start = time.time()
    df_denominator = df.groupby('state').count()   # getting counts of states
    # getting counts of every word in each state
    df_counts = df.groupby(['state', 'word']).size().reset_index()
    df_merged = df_counts.merge(df_denominator, left_on=[
                                'state'], right_on='state')  # merge
    df_merged = df_merged.rename(
        columns={"word_x": "word", 0: "word_count", "word_y": "state_count"})
    # get emission probability (count of word in that state/ state count)
    df_merged['Probability'] = df_merged.word_count/df_merged.state_count
    for index, row in tqdm(df_merged.iterrows()):  # for every known probabilty
        # append into the emission matrix
        emission_matrix.loc[row['state'], row['word']] = row['Probability']
    emission_matrix = emission_matrix.fillna(
        0)   # fill those null cells with zero
    # print(f'time elapsed {time.time()-start}')
    return emission_matrix


def emissionMatrix_special(df, emission_matrix):
    '''
    emissionMatrix_speical taking into account the special token with k=0.5
    Returns a matrix with the emission parameters matrix.
    df: the dataframe that contains the individual word instance and its label
    emission_matrix: The matrix that has been generated from create_matrix() that will be used to place
        the calculations
    '''
    k = 0.5
    df_denominator = df.groupby('state').count()   # getting counts of states
    # getting counts of every word in each state
    df_counts = df.groupby(['state', 'word']).size().reset_index()
    df_merged = df_counts.merge(df_denominator, left_on=[
                                'state'], right_on='state')  # merge
    df_merged = df_merged.rename(
        columns={"word_x": "word", 0: "word_count", "word_y": "state_count"})
    # get emission probability (count of word in that state/ state count)
    df_merged['Probability'] = df_merged.word_count/(df_merged.state_count+k)
    for index, row in tqdm(df_merged.iterrows()):  # for every known probabilty
        # append into the emission matrix
        emission_matrix.loc[row['state'], row['word']] = row['Probability']
    for i in df.state.unique().tolist():
        emission_matrix.loc[i, '#UNK#'] = float(k/df_denominator.loc[i]+k)
    emission_matrix = emission_matrix.fillna(
        0)   # fill those null cells with zero
    return emission_matrix

# getting the most probable states given the words from the emission matrix


def argmax(df):
    start = time.time()
    tags = {}
    for col in df.columns:
        tags[col] = df.index[df[col].argmax()]
    return tags

# tagging test df with most probable states


def tag_system(tags, df):
    test_ls = df['word'].tolist()
    tag_states = []
    for i in test_ls:
        if i in tags.keys():
            tag_states.append(tags[i])
        elif i == "":   # for blank lines, set state to be blank
            tag_states.append("")
        elif i not in tags.keys():
            tag_states.append(tags['#UNK#'])
    df['states'] = tag_states
    return df

# saving output as text file


def save_df(df, path):
    np.savetxt(path, df.values, fmt='%s', encoding="utf-8")
    # np.savetxt(r"EN\dev2_check.prediction", output.values, fmt='%s')


# file paths
# EN_train = 'EN/train'
# SG_train = 'SG/train'
# CN_train = 'CN/train'
# EN_test = 'EN/dev.in'
# SG_test = 'SG/dev.in'
# CN_test = 'CN/dev.in'
# EN_pred_2 = 'EN/dev_p2.pred'
# SG_pred_2 = 'SG/dev_p2.pred'
# CN_pred_2 = 'CN/dev_p2.pred'
# files = [['EN', EN_train, EN_test, EN_pred_2],
#          ['SG', SG_train, SG_test, SG_pred_2],
#          ['CN', CN_train, CN_test, CN_pred_2]]

# print('Starting Part 2')
# start_time = time.time()
# for i in files:
#     print(f'starting {i[0]}')
#     df_train = load_train(i[1])
#     df_test = load_test(i[2])
#     emission_matrix = createMatrix(df_train)
#     emission_matrix = emissionMatrix_special(df_train, emission_matrix)
#     tags = argmax(emission_matrix)
#     df_output = tag_system(tags, df_test)
#     save_df(df_output, i[3])
#     print(f'{i[0]} completed')

# print('Part 2 Complete')
# print(f'time elapsed {time.time()-start_time} seconds')


# def create_matrix(df):
#     new_df = pd.DataFrame(columns=df['state'].unique().tolist())
#     unique_words = df['word'].unique().tolist()
#     new_df['unique_words'] = unique_words
#     # Commented out for now
#     # new_df = new_df.set_index('unique_words')
#     return new_df

# def df_calculate(df, new_df):
#     '''
#     returns a matrix with the emission parameters matrix.
#     DF: the dataframe that contains the individual word instance and its label
#     NEW_DF: The matrix that has been generated from create_matrix() that will be used to place
#         the calculations
#     '''

#     unique_words = df['word'].unique().tolist()
#     count_all_states = df.groupby('state').count()
#     count = 0
#     for w in tqdm(unique_words):
#         word_df = [df.loc[df.word == w]][0]
#         # count for each state for each word
#         state_count = word_df.groupby('state').count()
#         for x in state_count.index:
#             numerator = (state_count['word'].loc[x])
#             denominator = (count_all_states['word'].loc[x])
#             new_df[x][count] = (numerator/denominator)
#         count += 1
#     new_df = new_df.fillna(0)
#     new_df = new_df.set_index(['unique_words'])
#     return new_df

# def df_calculate_2(df, new_df):
#     '''
#     df_calculate while taking into account the special token
#     Returns a matrix with the emission parameters matrix.
#     DF: the dataframe that contains the individual word instance and its label
#     NEW_DF: The matrix that has been generated from create_matrix() that will be used to place
#         the calculations
#     '''
#     unique_words = new_df['unique_words'].tolist()
#     count_all_states = df.groupby('state').count()
#     count = 0
#     for w in tqdm(unique_words):
#         word_df = [df.loc[df.word == w]][0]
#         # count for each state for each word
#         state_count = word_df.groupby('state').count()
#         for x in state_count.index:
#             numerator = (state_count['word'].loc[x])
#             denominator = (count_all_states['word'].loc[x])+0.5
#             new_df[x][count] = (numerator/denominator)
#         count += 1
#     unk_dict = {'unique_words': '#UNK#'}
#     for x in range(len(new_df.columns)-1):
#         unk_dict[new_df.columns[x]] = 0.5 / \
#             ((count_all_states['word'].loc[new_df.columns[x]])+0.5)
#     new_df = new_df.append(unk_dict, ignore_index=True)
#     new_df = new_df.fillna(0)
#     new_df = new_df.set_index(['unique_words'])
#     return new_df

# def argmax(new_df):
#     tags={}
#     for index, row in new_df.iterrows():
#         tags[index]=new_df.columns[row.argmax()]
#     return tags

# def tag_system(tags, testing_file):
#     df = load_test(testing_file)
#     test_ls = df['word'].tolist()

#     tag_states=[]
#     for i in test_ls:
#         if i in tags.keys():
#             tag_states.append(tags[i])
#         elif i=="":   # for blank lines, set state to be blank
#             tag_states.append("")
#         elif i not in tags.keys():
#             tag_states.append(tags['#UNK#'])

#     df['states']=tag_states
#     return df

import numpy as np
import pandas as pd
from tqdm import tqdm

EN_train = 'EN/train'
SG_train = 'SG/train'
CN_train = 'CN/train'
EN_test = 'EN/dev.in'
SG_test = 'SG/dev.in'
CN_test = 'CN/dev.in'

# load training file


def load_train(training_file):
    df = pd.read_csv(training_file, sep=' ',
                     header=None, error_bad_lines=False)
    df.columns = ['word', 'state']
    return df

# def load_test(test_file):
#     df = pd.read_csv(test_file, sep=' ', header = None, error_bad_lines=False)
#     df.columns=['word']
#     return df
def load_test(testing_file):
    ls = []
    f = open(testing_file)
    for line in f:
        ls.append(line.strip('\n'))
    df = pd.DataFrame(ls)
    df.columns=['word']
    return df

# load empty emission probability table
def create_matrix(df):
    new_df = pd.DataFrame(columns=df['state'].unique().tolist())
    unique_words = df['word'].unique().tolist()
    new_df['unique_words'] = unique_words
    # Commented out for now
    # new_df = new_df.set_index('unique_words')
    return new_df

# obtaining emission probabilities from training data
def df_calculate(df, new_df):
    '''
    returns a matrix with the emission parameters matrix.
    DF: the dataframe that contains the individual word instance and its label
    NEW_DF: The matrix that has been generated from create_matrix() that will be used to place
        the calculations
    '''

    unique_words = df['word'].unique().tolist()
    count_all_states = df.groupby('state').count()
    count = 0
    for w in tqdm(unique_words):
        word_df = [df.loc[df.word == w]][0]
        # count for each state for each word
        state_count = word_df.groupby('state').count()
        for x in state_count.index:
            numerator = (state_count['word'].loc[x])
            denominator = (count_all_states['word'].loc[x])
            new_df[x][count] = (numerator/denominator)
        count += 1
    new_df = new_df.fillna(0)
    new_df = new_df.set_index(['unique_words'])
    return new_df


def df_calculate_2(df, new_df):
    '''
    df_calculate while taking into account the special token
    Returns a matrix with the emission parameters matrix.
    DF: the dataframe that contains the individual word instance and its label
    NEW_DF: The matrix that has been generated from create_matrix() that will be used to place
        the calculations
    '''
    unique_words = new_df['unique_words'].tolist()
    count_all_states = df.groupby('state').count()
    count = 0
    for w in tqdm(unique_words):
        word_df = [df.loc[df.word == w]][0]
        # count for each state for each word
        state_count = word_df.groupby('state').count()
        for x in state_count.index:
            numerator = (state_count['word'].loc[x])
            denominator = (count_all_states['word'].loc[x])+0.5
            new_df[x][count] = (numerator/denominator)
        count += 1
    unk_dict = {'unique_words': '#UNK#'}
    for x in range(len(new_df.columns)-1):
        unk_dict[new_df.columns[x]] = 0.5 / \
            ((count_all_states['word'].loc[new_df.columns[x]])+0.5)
    new_df = new_df.append(unk_dict, ignore_index=True)
    new_df = new_df.fillna(0)
    new_df = new_df.set_index(['unique_words'])
    return new_df

def argmax(new_df):
    tag_dict={}
    for index, row in new_df.iterrows():
        tag_dict[index]=new_df.columns[row.argmax()]
    return tag_dict

def tag_system(tag_dict, testing_file):
    test_df = load_test(testing_file)
    test_ls = test_df['word'].tolist()

    tag_states=[]
    for i in test_ls:
        if i in tag_dict.keys():
            tag_states.append(tag_dict[i])
        elif i=="":   # for blank lines, set state to be blank
            tag_states.append("")
        elif i not in tag_dict.keys():
            tag_states.append(tag_dict['#UNK#'])

    test_df['states']=tag_states
    return test_df




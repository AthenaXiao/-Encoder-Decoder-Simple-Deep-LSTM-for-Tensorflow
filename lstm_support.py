from __future__ import print_function
import numpy as np
import tensorflow as tf
import collections

import nn_config

########
#
# METHOD: generate_text()
# DESCRIPTION: Translate LSTM output probabilities into target vocabulary using predefined dictionary.
#              *Avoid printing 'UNK' characters!*
# PARAMS:
#           prediction: array of probabilistic values that correspond to LSTM estimation outputs.
#           batch_size: number of parallel sequences processed.
#           length: MAX_LENGTH of each sequence w/ padding.
#           vocab_size: Size of target vocabulary - corresponds to number of classes within network estimations.
#           idx_to_word: Word ids mapped to corresponding words for translation.
# RETURNS:
#           batch_sequence: Batch of translated sentences.
#
########
def generate_text(prediction, batch_size, length, vocab_size, idx_to_word):

    batch_softmax = np.reshape(prediction, [batch_size, length, vocab_size])
    batch_sentence = []

    for sequence in batch_softmax:
        word_sequence = ''
        for char in sequence:
            vector_position = np.argmax(char)
            y_word = idx_to_word[vector_position]
            if y_word != 'ZERO':
                word_sequence = word_sequence + y_word + ' '
            else:
                word_sequence = word_sequence + ''
        batch_sentence.append(word_sequence)

    return batch_sentence

########
#
# METHOD: vectorize_data()
# DESCRIPTION: Transform sequences from word_id form into one_hot form.
# PARAMS:
#           word_sentences: sequences of word_ids
#           max_length: MAX_LENGTH of sentences w/ padding.
#           word_to_idx: data structure containing word mappings to their id representation.
# RETURNS:
#           sequences: Batch of one_hot sequences.
#
########
def vectorize_data(word_sentences, max_length, word_to_idx):
    sequences = np.zeros((len(word_sentences), max_length, len(word_to_idx)), dtype=float)
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences

########
#
# METHOD: load_data()
# DESCRIPTION: Load input and label data into algorithm to establish meta-data information
#              e.g. vocabulary size, word_id mapping etc.
# PARAMS:
#           inputs: directory and location for inputs .txt. file.
#           labels: directory and location for labels .txt. file.
#           vocabulary_size: define maximum vocabulary size for most commonly used words.
# RETURNS:
#           X_VOCAB_SIZE/Y_VOCAB_SIZE: Defined size for dictionary containing most common words (+ 'UNK' & 'ZERO')
#           x_word_to_idx/y_word_to_idx: Data structures mapping inputs/label words to id numbers.
#           x_idx_to_word/y_idx_to_word: Data structures mapping input/label id numbers to words.
#           X_MAX_LENGTH/Y_MAX_LENGTH: MAX_LENGTH of sentences (data to be used for sentence padding).
#           X_DATA_LENGTH: Number of Inputs/Labels
#
########
def load_data(inputs = nn_config.inputs, labels = nn_config.labels, vocabulary_size = nn_config.vocabulary_size):

    print('[INFO] Importing RAW Inputs and Labels...')
    x_data_source = open(inputs, 'r')
    y_data_source = open(labels, 'r')

    # Feed input data in backwards for better translation performance.
    x_data = (tf.compat.as_str(x_data_source.read()).lower().split('\n'))
    x_data = [x_data[_].split(' ')[::-1] for _ in range(len(x_data))]
    X_DATA_SIZE = len(x_data)
    print('[NOTIFICATION] Here is a snippet of the Inputs!')
    print(x_data[0:2])

    y_data = (tf.compat.as_str(y_data_source.read()).lower().split('\n'))
    y_data = [y_data[_].split(' ') for _ in range(len(y_data))]
    Y_DATA_SIZE = len(y_data)
    print('[NOTIFICATION] Here is a snippet of the Labels!')
    print(y_data[0:2])


    print('[INFO] Creating Dictionary Indexes...')
    #Word Dictionary
    x_word_list = []
    y_word_list = []
    [x_word_list.extend(x_data[_]) for _ in range(len(x_data))]
    [y_word_list.extend(y_data[_]) for _ in range(len(y_data))]

    x_word_dictionary = []
    y_word_dictionary = []
    print('[INFO] Finding most common vocabulary...')
    print('[INFO] The common vocabulary threshold is set to: {}'.format(nn_config.vocabulary_size))
    x_word_dictionary.extend(collections.Counter(x_word_list).most_common(vocabulary_size))
    y_word_dictionary.extend(collections.Counter(y_word_list).most_common(vocabulary_size))

    print('[INFO] Clearing up memory...')
    del x_word_list, y_word_list

    print('[INFO] Mapping vocabulary to idx...')
    #word/idx mapping
    x_idx_to_word = [word[0] for idx, word in enumerate(x_word_dictionary)]
    x_idx_to_word.insert(0, 'ZERO')
    x_idx_to_word.append('UNK')
    print('[NOTIFICATION] Here is a snippet of the Input Dictionary!')
    print(x_idx_to_word[0:20])

    y_idx_to_word = [word[0] for idx, word in enumerate(y_word_dictionary)]
    y_idx_to_word.insert(0, 'ZERO')
    y_idx_to_word.append('UNK')
    print('[NOTIFICATION] Here is a snippet of the Label Dictionary!')
    print(y_idx_to_word[0:20])

    x_word_to_idx = {word:ix for ix, word in enumerate(x_idx_to_word)}
    y_word_to_idx = {word: ix for ix, word in enumerate(y_idx_to_word)}

    X_VOCAB_SIZE = len(x_word_dictionary) + 2
    Y_VOCAB_SIZE = len(y_word_dictionary) + 2

    print('[INFO] Converting vocabulary to index value...')
    # Converting each word to its index value
    for i, sentence in enumerate(x_data):
        for j, word in enumerate(sentence):
            if word in x_word_to_idx:
                x_data[i][j] = x_word_to_idx[word]
            else:
                x_data[i][j] = x_word_to_idx['UNK']
    print('[INFO] Input conversion complete!')
    print('[NOTIFICATION] Here is a snippet of the Input Index Values!')
    print(x_data[0:2])

    for i, sentence in enumerate(y_data):
        for j, word in enumerate(sentence):
            if word in y_word_to_idx:
                y_data[i][j] = y_word_to_idx[word]
            else:
                y_data[i][j] = y_word_to_idx['UNK']
    print('[INFO] Label conversion complete!')
    print('[NOTIFICATION] Here is a snippet of the Label Index Values!')
    print(y_data[0:2])

    X_MAX_LENGTH = max([len(x_data[_]) for _ in range(len(x_data))])
    Y_MAX_LENGTH = max([len(y_data[_]) for _ in range(len(y_data))])

    del x_data, y_data

    return X_VOCAB_SIZE, Y_VOCAB_SIZE, x_idx_to_word, x_word_to_idx, y_idx_to_word, y_word_to_idx, X_MAX_LENGTH, Y_MAX_LENGTH, X_DATA_SIZE

########
#
# METHOD: load_data_by_batch()
# DESCRIPTION: Load input and label data into algorithm to establish content.  Do by batch_size to conserve memory.
# PARAMS:
#           batch_number: keep track of progress throughout the document.
#           batch_size: number of parallel sequences to be processed.
#           x_word_to_idx/y_word_to_idx: word to idx mapping to create sequence of idxs.
#           inputs: directory and location for inputs .txt. file.
#           labels: directory and location for labels .txt. file.
# RETURNS:
#           x_data/y_data: Returns sequences of word idxs for processing and padding.
#
########
def load_data_by_batch(batch_number, x_word_to_idx, y_word_to_idx, inputs = nn_config.inputs, labels = nn_config.labels,
                       batch_size = nn_config.batch_size):

    start_batch = batch_number * batch_size
    end_batch = start_batch + batch_size

    print('\n[INFO] Importing RAW Inputs and Labels for batch...')
    x_data_source = open(inputs, 'r')
    y_data_source = open(labels, 'r')

    #Feed input data in backwards for better translation performance.
    x_data = (tf.compat.as_str(x_data_source.read()).lower().split('\n'))
    x_data = [x_data[_].split(' ')[::-1] for _ in range(len(x_data))]
    x_data = x_data[start_batch:end_batch]

    y_data = (tf.compat.as_str(y_data_source.read()).lower().split('\n'))
    y_data = [y_data[_].split(' ') for _ in range(len(y_data))]
    y_data = y_data[start_batch:end_batch]

    # Converting each word to its index value
    for i, sentence in enumerate(x_data):
        for j, word in enumerate(sentence):
            if word in x_word_to_idx:
                x_data[i][j] = x_word_to_idx[word]
            else:
                x_data[i][j] = x_word_to_idx['UNK']

    for i, sentence in enumerate(y_data):
        for j, word in enumerate(sentence):
            if word in y_word_to_idx:
                y_data[i][j] = y_word_to_idx[word]
            else:
                y_data[i][j] = y_word_to_idx['UNK']

    return x_data, y_data
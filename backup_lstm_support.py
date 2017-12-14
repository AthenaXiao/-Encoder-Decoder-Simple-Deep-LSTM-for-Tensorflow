from __future__ import print_function
import numpy as np
import tensorflow as tf

# Method for generating text
def generate_text_keras(model, length, vocab_size, ix_to_char):
	# starting with random character
	ix = [np.random.randint(vocab_size)]
	y_char = [ix_to_char[ix[-1]]]
	X = np.zeros((1, length, vocab_size))
	for i in range(length):
		# appending the last predicted character to sequence
		X[0, i, :][ix[-1]] = 1
		print(ix_to_char[ix[-1]], end="")
		ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
		y_char.append(ix_to_char[ix[-1]])
	return ('').join(y_char)

def generate_text(prediction, batch_size, length, vocab_size, idx_to_char):


    batch_softmax = np.reshape(prediction, [batch_size, length, vocab_size])

    batch_sequence = []

    for sequence in batch_softmax:
        char_sequence = ''
        for char in sequence:
            vector_position = np.argmax(char)
            y_char = idx_to_char[vector_position]
            char_sequence = char_sequence + y_char
        batch_sequence.append(char_sequence)


    return batch_sequence


#Method for preparing data.
def load_data(seq_len):

    data_source = open('data_files/data_source.txt', 'r')

    #Data holder contains sequence of chars
    data = np.array(list(tf.compat.as_str(data_source.read().encode('utf8'))))
    DATA_SIZE = len(data)
    #Chars Dictionary
    char_dictionary = np.unique(data)
    VOCAB_SIZE = len(char_dictionary)

    #Character/idx mapping
    idx_to_char = {ix:char for ix, char in enumerate(char_dictionary)}
    char_to_idx = {char:ix for ix, char in enumerate(char_dictionary)}

    LENGTH_OF_SEQUENCE = seq_len
    NUMBER_OF_FEATURES = VOCAB_SIZE
    NUMBER_OF_SEQUENCES = DATA_SIZE/LENGTH_OF_SEQUENCE

    inputs = np.zeros((NUMBER_OF_SEQUENCES, LENGTH_OF_SEQUENCE, NUMBER_OF_FEATURES))
    labels = np.zeros((NUMBER_OF_SEQUENCES, LENGTH_OF_SEQUENCE, NUMBER_OF_FEATURES))

    for i in range(0, NUMBER_OF_SEQUENCES):
        X_sequence = data[i*LENGTH_OF_SEQUENCE:(i+1)*LENGTH_OF_SEQUENCE]
        X_sequence_ix = [char_to_idx[value] for value in X_sequence]
        input_sequence = np.zeros((LENGTH_OF_SEQUENCE, VOCAB_SIZE))
        for j in range(LENGTH_OF_SEQUENCE):
            input_sequence[j][X_sequence_ix[j]] = 1.
        inputs[i] = input_sequence

        y_sequence = data[i*LENGTH_OF_SEQUENCE+1:(i+1)*LENGTH_OF_SEQUENCE+1]
        y_sequence_ix = [char_to_idx[value] for value in y_sequence]
        target_sequence = np.zeros((LENGTH_OF_SEQUENCE, VOCAB_SIZE))
        for j in range(LENGTH_OF_SEQUENCE):
            target_sequence[j][y_sequence_ix[j]] = 1.
        labels[i] = target_sequence

    return inputs, labels, VOCAB_SIZE, idx_to_char
import tensorflow as tf
import numpy as np
import lstm_support
import nn_config

########
#
# INITIALIZE VARIABLES & HYPER-PARAMETERS FOR LSTM MODEL
#
########

logs_path = nn_config.logs_path

print('[INFO] Loading data from {} and {}...'.format(nn_config.inputs, nn_config.labels))
X_VOCAB_SIZE, Y_VOCAB_SIZE, x_idx_to_word, x_word_to_idx, y_idx_to_word, y_word_to_idx, X_MAX_LENGTH, Y_MAX_LENGTH, X_DS_LENGTH = lstm_support.load_data()
print('[INFO] Data load complete!\n')

batch_size = nn_config.batch_size
num_batches = X_DS_LENGTH/batch_size
num_epochs = nn_config.num_epochs

state_size = nn_config.state_size
encoder_layers = nn_config.encoder_layers
decoder_layers = nn_config.decoder_layers

encoder_x = tf.placeholder(dtype=tf.int32, shape=[None, None]) #[batch_size, X_MAX_LENGTH]
decoder_x = tf.placeholder(dtype=tf.int32, shape=[None, None, Y_VOCAB_SIZE]) #[batch_size, Y_MAX_LENGTH, Y_VOCAB_SIZE]
y = tf.placeholder(dtype=tf.float32, shape=[None, None, Y_VOCAB_SIZE])#[batch_size, Y_MAX_LENGTH, Y_VOCAB_SIZE]
init_state = tf.placeholder(tf.float32, [encoder_layers, 2, batch_size, state_size])

print('[CONFIG] LSTM Sequence-to-Sequence Translation Model Configuration:')
print('[CONFIG] Batch size: {}'.format(batch_size))
print('[CONFIG] Number of Hidden Layers (Encoder/Decoder): {}/{}'.format(encoder_layers, decoder_layers))
print('[CONFIG] Hidden Layer State Size: {}'.format(state_size))
print('[CONFIG] X Placeholder Shape: [{}, {}]'.format('None', 'None'))
print('[CONFIG] Y Placeholder Shape: [{}, {}, {}]\n').format('None', Y_MAX_LENGTH, Y_VOCAB_SIZE)

########
#
# METHOD: variable_summaries()
# DESCRIPTION: Create histogram and distribution logging data for TensorFlow Variables - can be viewed in TensorBoard.
# PARAMS:
#           var: TensorFlow variable to create logging data for.
# RETURNS:
#           None
#
########
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

########
#
# METHOD: lstm_model()
# DESCRIPTION: Create TensorFlow LSTM Model to process our data in the Encoder/Decoder Architecture:
#              NOTE: Attention is not yet implemented!!
# PARAMS:
#           data: x TensorFlow placeholder data.
# RETURNS:
#           decoder_final_state: Final state for Decoder Network.
#           logits: Un-normalised logits to pass to loss function.
#           labels: y TensorFlow placeholder data.
#           prediction: Softmax prediction output for translation.
#
########
def lstm_model(data):

    with tf.device('/cpu:0'):

        with tf.variable_scope('encoder_word_embeddings'):

            word_embeddings = tf.get_variable('encoder_word_embeddings', [X_VOCAB_SIZE, nn_config.embedding_size])
            encoder_embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, encoder_x)
            encoder_embedded_word_ids = tf.reshape(encoder_embedded_word_ids, [-1, X_MAX_LENGTH, nn_config.embedding_size])

        with tf.variable_scope('encoder'):
            ####
            #
            # LSTM ENCODER
            #
            ####
            # Forward passes
            state_per_layer_list = tf.unstack(init_state, axis=0)
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                for idx in range(encoder_layers)])

            encoder_stacked_cell = []

            for _ in range(encoder_layers):
                encoder_single_cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
                if _ == 0:
                    with tf.name_scope('encoder_dropout') as scope:
                        encoder_single_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_single_cell,
                                                        output_keep_prob=0.75)  # add dropout to first LSTM layer only.
                encoder_stacked_cell.append(encoder_single_cell)

            encoder_cell = tf.nn.rnn_cell.MultiRNNCell(encoder_stacked_cell, state_is_tuple=True)

            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                             inputs=encoder_embedded_word_ids,
                                                             initial_state=rnn_tuple_state)

            del encoder_outputs, encoder_cell, state_per_layer_list, encoder_stacked_cell

        with tf.variable_scope('decoder_word_embeddings'):

            word_embeddings = tf.get_variable('decoder_word_embeddings', [Y_VOCAB_SIZE, nn_config.embedding_size])
            decoder_embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, decoder_x)
            decoder_embedded_word_ids = tf.reshape(decoder_embedded_word_ids,
                                                        [-1, Y_MAX_LENGTH, Y_VOCAB_SIZE * nn_config.embedding_size])

        with tf.variable_scope('decoder'):
            ####
            #
            # LSTM DECODER
            #
            ####
            decoder_stacked_cell = []
            for _ in range(decoder_layers):
                decoder_single_cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
                if _ == 0:
                    with tf.name_scope('decoder_dropout') as scope:
                        decoder_single_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_single_cell,
                                                    output_keep_prob=0.75)  # add dropout to first LSTM layer only.
                decoder_stacked_cell.append(decoder_single_cell)

            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_stacked_cell, state_is_tuple=True)

            decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(cell=decoder_cell,
                                                             inputs=decoder_embedded_word_ids,
                                                             initial_state=encoder_final_state)

            outputs = tf.reshape(decoder_outputs, [-1, state_size])

            with tf.name_scope('decoder_hidden_states') as scope:
                W2 = tf.Variable(tf.random_normal([state_size, Y_VOCAB_SIZE]), dtype=tf.float32)
                variable_summaries(W2)
                b2 = tf.Variable(tf.zeros([1, Y_VOCAB_SIZE]), dtype=tf.float32)
                variable_summaries(b2)

                logits = tf.matmul(outputs, W2) + b2 # Broadcasted addition
                tf.summary.histogram('pre_activations', logits)
                prediction = tf.nn.softmax(logits)
                tf.summary.histogram('activations', prediction)
            labels = y

        return decoder_final_state, labels , logits, prediction


########
#
# METHOD: config_tensorflow_hardware()
# DESCRIPTION: Configure CPU to utilize all available threads to workload.
# PARAMS:
#           None
# RETURNS:
#           config: config Protobuf
#
########
def config_tensorflow_hardware():

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 4  # No. physical cores
    config.inter_op_parallelism_threads = 4  # No. physical cores

    return config

########
#
# METHOD: lstm_train()
# DESCRIPTION: Feed the network model using the training corpus data.  Optimize the results using 'AdamOptimizer'.
#              Print the results every 5 batches to visualise learning.
#              Write all variable changes to TensorBoard Protobuf for Variable and Network visualization.
# PARAMS:
#           None
# RETURNS:
#           None
#
########
def lstm_train():

    #Initialize CPU config..
    config = config_tensorflow_hardware()

    current_state, labels, logits, prediction = lstm_model(encoder_x)

    with tf.name_scope('cross_entropy') as scope:
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        total_cost = tf.reduce_mean(cost)
        tf.summary.scalar('cross-entropy', total_cost)

    with tf.name_scope('optimizer') as scope:
        optimizer = tf.train.AdamOptimizer(nn_config.learning_rate).minimize(total_cost)

    #Keep previous 30 model checkpoints..
    saver = tf.train.Saver(max_to_keep=30)

    with tf.Session(config=config) as sess:

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

        sess.run(tf.global_variables_initializer())

        epoch = 0

        if nn_config.restore:
            print('Loading variables from {}'.format(nn_config.ckpt_model_directory + nn_config.meta_file + nn_config.meta_num + nn_config.meta_ext))
            saver.restore(sess, nn_config.ckpt_model_directory + nn_config.meta_file + nn_config.meta_num)

        _current_state = np.zeros((encoder_layers, 2, batch_size, state_size))

        while True: #Unlimited epochs for Training

            epoch_loss = 0

            for batch in range(num_batches):

                batch_x, batch_y = lstm_support.load_data_by_batch(batch, x_word_to_idx, y_word_to_idx)

                batch_loss = 0

                print('[INFO] Zero padding RAW batch data...'.format(batch))
                for _ in range(len(batch_x)):
                    x_length = len(batch_x[_])
                    y_length = len(batch_y[_])

                    batch_x[_].extend(np.zeros([X_MAX_LENGTH - x_length], dtype=int))
                    batch_y[_].extend(np.zeros([Y_MAX_LENGTH - y_length], dtype=int))

                y_one_hot = lstm_support.vectorize_data(batch_y, Y_MAX_LENGTH, y_word_to_idx)

                del batch_y

                print('[INFO] Passing batch values into Encoder/Decoder for LSTM network optimization...'.format(batch))

                summary, _total_cost, _train_step, _current_state, _prediction_series = sess.run(
                [merged, total_cost, optimizer, current_state, prediction],
                feed_dict={
                        encoder_x: batch_x,
                        decoder_x: y_one_hot,
                        y: y_one_hot,
                        init_state: _current_state
                    })

                writer.add_summary(summary, epoch)

                if batch % 2 == 0:

                    print('[INFO] Translating Softmax Probabilities...')

                    text = lstm_support.generate_text(_prediction_series, batch_size, Y_MAX_LENGTH, Y_VOCAB_SIZE, y_idx_to_word)

                    if epoch % 5 == 0:
                        save_output_path = 'data_files/training/encoder-decoder-output/TRAINING_TRANSLATION_OUTPUT_EPOCH_{}_STATE-SIZE_{}_NUM-LAYERS_{}_LEARNING-RATE{}_EMBEDDING-SIZE_{}.txt'.format(epoch, state_size, decoder_layers, nn_config.learning_rate, nn_config.embedding_size)

                        with open(save_output_path,'wb') as f:
                         for item in text:
                             f.write("%s\n" % item)
                        print('[NOTIFICATION] Translation file saved to: {}'.format(save_output_path))

                    print('[INFO]: Batch {} optimized output for Epoch {}:'.format(batch, epoch))
                    for _ in range(batch_size):
                        print(text[_])

                batch_loss += _total_cost
                epoch_loss += _total_cost

                print('\n[STATUS] Batch {}/{} complete! Batch Loss: {}'.format(batch, num_batches, batch_loss))

            if epoch % 1 == 0:

                save_model_path = "data_files/training/saved_models/translation_model"
                if epoch == 0:
                    save_path = saver.save(sess, save_model_path, epoch)
                else:
                    save_path = saver.save(sess, save_model_path, epoch, write_meta_graph=False)

                print("[STATUS] Model ckpt saved in file: %s" % save_path)

            print('[STATUS] Epoch {} complete! Epoch Loss: {}'.format(epoch, epoch_loss))

            saver.save(sess, logs_path + '/model.ckpt', epoch)
            epoch+=1

lstm_train()
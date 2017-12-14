import tensorflow as tf
import numpy as np
import lstm_support

SEQ_LEN = 'NONE'


print('How much would you like me to remember while learning?\n')

while SEQ_LEN.isdigit() == False:
    SEQ_LEN = raw_input('ENTER [SEQUENCE_LENGTH] NUMBER:')
    if (SEQ_LEN).isdigit() == False:
        print('You must provide me with an integer! Integer Examples: [2, 6, 5, 74, 674, 2678, 123445]\n')

SEQ_LEN = int(SEQ_LEN)

inputs, targets, VOCAB_SIZE, idx_to_char = lstm_support.load_data(SEQ_LEN)

batch_size = 128
num_batches = len(inputs)/batch_size
num_epochs = 200

state_size = 256  #64 was a good number for linux OS!
rnn_num_hidden_layers = 3


x = tf.placeholder(dtype=tf.float32, shape=[None, SEQ_LEN, VOCAB_SIZE])
y = tf.placeholder(dtype=tf.float32, shape=[None, SEQ_LEN, VOCAB_SIZE])
init_state = tf.placeholder(dtype=tf.float32, shape=[rnn_num_hidden_layers, 2, batch_size, state_size])

def lstm_model(data):

    with tf.device('/cpu:0'):
            # Forward passes

            cell = tf.nn.rnn_cell.LSTMCell(state_size)

            states_series, current_state = tf.nn.dynamic_rnn(cell=cell,
                                                             inputs=x,
                                                             dtype=tf.float32)

            states_series = tf.reshape(states_series, [-1, state_size])

            W2 = tf.Variable(tf.random_normal([state_size, VOCAB_SIZE]), dtype=tf.float32)
            b2 = tf.Variable(tf.random_normal([1, VOCAB_SIZE]), dtype=tf.float32)

            logits = tf.matmul(states_series, W2) + b2  # Broadcasted addition

            labels = y

            return current_state, labels , logits

def config_tensorflow_hardware():

    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 2  # No. physical cores
    config.inter_op_parallelism_threads = 2  # No. physical cores

    return config

def lstm_train(inputs, targets):

    config = config_tensorflow_hardware()

    labels = []
    logits = []

    current_state, labels, logits = lstm_model(x)

    costs = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    total_cost = tf.reduce_mean(costs)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(total_cost)

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        epoch = 1

        #while epoch <= num_epochs:

        while True: #Unlimited epochs for Training

            epoch_loss = 0

            _current_state = np.zeros((rnn_num_hidden_layers, 2, batch_size, state_size))

            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size

                batch_loss = 0

                batch_x = inputs[start_idx:end_idx]
                batch_y = targets[start_idx:end_idx]

                _total_cost, _train_step = sess.run(
                [total_cost, optimizer],
                feed_dict={
                        x: batch_x,
                        y: batch_y,
                    })

                if batch == 0:
                    prediction = tf.nn.softmax(logits)
                    feed_dict = {x: batch_x}

                    batch_word_prediction = np.array(sess.run(prediction, feed_dict=feed_dict))

                    text = lstm_support.generate_text(batch_word_prediction, batch_size, SEQ_LEN, VOCAB_SIZE, idx_to_char)

                    with open('data_files/training/multi_lstm_output_generations/training_text_output_epoch_{}_state_size_{}.txt'.format(epoch, state_size),'wb') as f:
                        for item in text:
                            f.write("%s\n" % item)

                    for _ in range(32):
                        print(text[_])

                batch_loss += _total_cost
                epoch_loss += _total_cost

                print('Batch', batch, 'completed out of', num_batches, 'loss:', batch_loss)
            print('Epoch', epoch, 'completed out of', num_epochs, 'loss:', epoch_loss)
            epoch+=1

lstm_train(inputs, targets)
########
#
# INPUTS & LABELS FOR LSTM NETWORK MACHINE TRANSLATION
# NOTE: THIS SHOULD BE WITHIN THE DATA_FILES FOLDER!
#
########
#inputs = 'data_files/europarl-v7.en.txt'
#labels = 'data_files/europarl-v7.de.txt'
inputs = 'data_files/input/europarl-v7.de.txt'
labels = 'data_files/input/europarl-v7.en.txt'

val_inputs = None
val_labels = None

test_inputs = None
test_labels = None

logs_path = 'tmp/logs'

########
#
# VOCABULARY CONFIGURATION
# NOTE: THIS IS THE MAXIMUM VOCABULARY SIZE, ALL OTHER WORDS ARE 'UNK' - Adjust based on RAM allocation!
#
########
vocabulary_size = 10000

########
#
# DATA SET CONFIGURATION
# NOTE: FLOAT VALUES ARE PERCENTAGE OF DATA FILES
#
########
training = 0.75
validation = 0.15
test = 0.10

########
#
# LSTM MODEL CONFIGURATION (TENSORFLOW)
#
########
batch_size = 8
num_epochs = 200
state_size = 128  #64, 128, 256 was a good number for linux OS!
number_of_layers = 3
encoder_layers = number_of_layers
decoder_layers = number_of_layers
learning_rate = 0.001
learning_rate_decay = 0.1
embedding_size = 128

########
#
# LSTM MODEL SAVE/RESTORE CONFIGURATION
#
########
ckpt_model_directory = 'data_files/training/saved_models'
meta_num = '2'
meta_ext = '.meta'
meta_file = '/translation_model-'
restore = False
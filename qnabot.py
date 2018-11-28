from __future__ import print_function, division
from builtins import range, input


from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import random

import keras.backend as K
if len(K.tensorflow_backend._get_available_gpus()) > 0:
  from keras.layers import CuDNNLSTM as LSTM
  from keras.layers import CuDNNGRU as GRU


BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 100  # Number of epochs to train for.
LATENT_DIM = 256  # Latent dimensionality of the encoding space.
NUM_SAMPLES = 2000  # Number of samples to train on.
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100


questions = []
answers_texts = []
answers_texts_inputs = []

flag = False
"""
t = 0
for line in open('movie_lines.txt',encoding='utf8', errors='ignore'):
    if (t>NUM_SAMPLES):
        break
    data = line.strip().split('+++$+++')
    data = data[len(data)-1]
    t += 1
    if not flag:
        questions.append(data)
        flag = True
    else:
        answer_text = data + ' <eos>'
        answer_text_input = '<sos> ' + data
        answers_texts.append(answer_text)
        answers_texts_inputs.append(answer_text_input)
        questions.append(data)
    
questions = questions[:-1]
"""

dataset = pd.read_csv('rdany_chat.csv')
dataset = list(dataset['text'])
t=0
for l in dataset:
   if t>NUM_SAMPLES:
       break
   t+=1
   if flag:
       answer_text =l+' <eos>'
       answer_text_input = '<sos> '+l
       answers_texts.append(answer_text)
       answers_texts_inputs.append(answer_text_input)
       questions.append(l)
   else:
       questions.append(l)
       flag = True

questions = questions[:-1]

    
# tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_inputs.fit_on_texts(questions)
question_sequences = tokenizer_inputs.texts_to_sequences(questions)

# get the word to index mapping for input language
word2idx_questions = tokenizer_inputs.word_index
print('Found %s unique input tokens.' % len(word2idx_questions))

# determine maximum length input sequence
max_len_question = max(len(s) for s in question_sequences)

# tokenize the outputs
# don't filter out special characters
# otherwise <sos> and <eos> won't appear
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(answers_texts + answers_texts_inputs) # inefficient, oh well
answer_sequences = tokenizer_outputs.texts_to_sequences(answers_texts)
answer_sequences_inputs = tokenizer_outputs.texts_to_sequences(answers_texts_inputs)

# get the word to index mapping for output language
word2idx_answers = tokenizer_outputs.word_index
print('Found %s unique output tokens.' % len(word2idx_answers))

# store number of output words for later
# remember to add 1 since indexing starts at 1
num_words_answers = len(word2idx_answers) + 1

# determine maximum length output sequence
max_len_answer = max(len(s) for s in answer_sequences)


# pad the sequences
encoder_inputs = pad_sequences(question_sequences, maxlen=max_len_question)
print("encoder_inputs.shape:", encoder_inputs.shape)
print("encoder_inputs[0]:", encoder_inputs[0])

decoder_inputs = pad_sequences(answer_sequences_inputs, maxlen=max_len_answer, padding='post')
print("decoder_inputs[0]:", decoder_inputs[0])
print("decoder_inputs.shape:", decoder_inputs.shape)

decoder_targets = pad_sequences(answer_sequences, maxlen=max_len_answer, padding='post')


# store all the pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open('glove.6B.%sd.txt' % EMBEDDING_DIM,encoding='utf8') as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))



# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_NUM_WORDS, len(word2idx_questions) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx_questions.items():
  if i < MAX_NUM_WORDS:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector




# create embedding layer
embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  input_length=max_len_question,
  #trainable=True
)


# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
decoder_targets_one_hot = np.zeros(
  (
    len(questions),
    max_len_answer,
    num_words_answers
  ),
  dtype='float32'
)

# assign the values
for i, d in enumerate(decoder_targets):
  for t, word in enumerate(d):
    decoder_targets_one_hot[i, t, word] = 1




##### build the model #####
encoder_inputs_placeholder = Input(shape=(max_len_question,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(
  LATENT_DIM,
  return_state=True,
  # dropout=0.5 # dropout not available on gpu
)
encoder_outputs, h, c = encoder(x)
# encoder_outputs, h = encoder(x) #gru

# keep only the states to pass into decoder
encoder_states = [h, c]
# encoder_states = [state_h] # gru

# Set up the decoder, using [h, c] as initial state.
decoder_inputs_placeholder = Input(shape=(max_len_answer,))

# this word embedding will not use pre-trained vectors
# although you could
decoder_embedding = Embedding(num_words_answers, LATENT_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

# since the decoder is a "to-many" model we want to have
# return_sequences=True
decoder_lstm = LSTM(
  LATENT_DIM,
  return_sequences=True,
  return_state=True,
  # dropout=0.5 # dropout not available on gpu
)
decoder_outputs, _, _ = decoder_lstm(
  decoder_inputs_x,
  initial_state=encoder_states
)

# decoder_outputs, _ = decoder_gru(
#   decoder_inputs_x,
#   initial_state=encoder_states
# )

# final dense layer for predictions
decoder_dense = Dense(num_words_answers, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Create the model object
model = Model([encoder_inputs_placeholder, decoder_inputs_placeholder], decoder_outputs)

# Compile the model and train it
model.compile(
  optimizer='rmsprop',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)
r = model.fit(
  [encoder_inputs, decoder_inputs], decoder_targets_one_hot,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=0.2,
)

# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

# Save model
model.save('s2s.h5')




##### Make predictions #####
# As with the poetry example, we need to create another model
# that can take in the RNN state and previous word as input
# and accept a T=1 sequence.

# The encoder will be stand-alone
# From this we will get our initial decoder hidden state
encoder_model = Model(encoder_inputs_placeholder, encoder_states)

decoder_state_input_h = Input(shape=(LATENT_DIM,))
decoder_state_input_c = Input(shape=(LATENT_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_states_inputs = [decoder_state_input_h] # gru

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# this time, we want to keep the states too, to be output
# by our sampling model
decoder_outputs, h, c = decoder_lstm(
  decoder_inputs_single_x,
  initial_state=decoder_states_inputs
)
# decoder_outputs, state_h = decoder_lstm(
#   decoder_inputs_single_x,
#   initial_state=decoder_states_inputs
# ) #gru
decoder_states = [h, c]
# decoder_states = [h] # gru
decoder_outputs = decoder_dense(decoder_outputs)

# The sampling model
# inputs: y(t-1), h(t-1), c(t-1)
# outputs: y(t), h(t), c(t)
decoder_model = Model(
  [decoder_inputs_single] + decoder_states_inputs, 
  [decoder_outputs] + decoder_states
)

# map indexes back into real words
# so we can view the results
idx2word_questions = {v:k for k, v in word2idx_questions.items()}
idx2word_answers = {v:k for k, v in word2idx_answers.items()}


def decode_sequence(input_seq):
  # Encode the input as state vectors.
  states_value = encoder_model.predict(input_seq)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1))

  # Populate the first character of target sequence with the start character.
  # NOTE: tokenizer lower-cases all words
  target_seq[0, 0] = word2idx_answers['<sos>']

  # if we get this we break
  eos = word2idx_answers['<eos>']

  # Create the translation
  output_sentence = []
  for _ in range(max_len_answer):
    output_tokens, h, c = decoder_model.predict(
      [target_seq] + states_value
    )
    # output_tokens, h = decoder_model.predict(
    #     [target_seq] + states_value
    # ) # gru

    # Get next word
    idx = np.argmax(output_tokens[0, 0, :])

    # End sentence of EOS
    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = idx2word_answers[idx]
      output_sentence.append(word)

    # Update the decoder input
    # which is just the word just generated
    target_seq[0, 0] = idx

    # Update states
    states_value = [h, c]
    # states_value = [h] # gru

  return ' '.join(output_sentence)



while True:
  # Do some test translations
  i = np.random.choice(len(questions))
  input_seq = encoder_inputs[i:i+1]
  translation = decode_sequence(input_seq)
  print('-')
  print('Input:', questions[i])
  print('Translation:', translation)

  ans = input("Continue? [Y/n]")
  if ans and ans.lower().startswith('n'):
    break

test_sentences = ['do you know me?']
test_sequences = tokenizer_inputs.texts_to_sequences(test_sentences)
test_sequences = pad_sequences(test_sequences, maxlen=max_len_question)
translation = decode_sequence(test_sequences)
print('Translation: ', translation)

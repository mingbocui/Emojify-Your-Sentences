# USIng LSTM in keras
import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')
maxLen = len(max(X_train, key=len).split())#because we have to embed 0 for other short sentences

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def sentences_to_indices(X, word_to_index, max_len):
	m = X.shape[0]
	X_indices = np.zeros([m, max_len])
	for i in range(m):
		sentence_words = (X[i].lower()).split()
		j = 0
		for w in sentence_words:
			X_indices[i, j] = word_to_index[w]
			j = j + 1

	return X_indices
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
	vocab_len = len(word_to_index) + 1
	emb_dim = word_to_vec_map["cucumber"].shape[0]
	emb_matrix = np.zeros([vocab_len, emb_dim])
	for word, index in word_to_index.items():
		emb_matrix[index, :] = word_to_vec_map[word]
	#define keras embedding layer
	embedding_layer = Embedding(vocab_len, emb_dim, 
		embeddings_initializer='uniform', trainable = False)
	# has to use this build before set_weights
	embedding_layer.build((None,))
	# kind of initialization
	embedding_layer.set_weights([emb_matrix])

	return embedding_layer

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
	sentence_indices = Input(shape=input_shape, dtype='int32')
	embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
	embeddings = embedding_layer(sentence_indices)

	# Be careful, the returned output should be a single hidden state, not a batch of sequences.
	#在输出序列中，返回单个 hidden state值还是返回全部time step 的 hidden state值。 False 返回单个， true 返回全部。
	# 128：dimensionality of the output space.
	# if return_sequence=True, it will output all hidden states
	X = LSTM(128, return_sequences=True)(embeddings)
	X = Dropout(0.5)(X)
	X = LSTM(128, return_sequences=False)(X)
	X = Dropout(0.5)(X)
	X = Dense(5)(X)
	X = Activation("softmax")(X)

	model = Model(inputs=sentence_indices, outputs=X)

	return model
#max_len = 5

model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary

from keras.utils import plot_model
import matplotlib.pyplot as plt
plot_model(model, to_file='model.png')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)


model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle = True)
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

#test
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test_indices)):
	x = X_test_indices
	num = np.argmax(pred[i])
	if(num != Y_test[i]):
		print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())

        
####################### TEST PART ############################################
x_test = np.array(['not feeling love'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
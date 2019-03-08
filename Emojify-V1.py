import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix

# Baseline model
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')
maxLen = len(max(X_train, key=len).split())#because we have to embed 0 for other short sentences

Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def sentence_to_avg(sentence, word_to_vec_map):
	words = (sentence.lower()).split()
	avg = np.zeros([50,])#because word_to_vec_map is 50-dimensional
	for w in words:
		avg += word_to_vec_map[w]
		avg = avg/len(words)

	return avg

def model(X, Y, word_to_vec_map, learning_rate = 0.01, num_iterations = 400):
	np.random.seed(1)
	m = Y.shape[0]
	n_y = 5
	n_h = 50

	W = np.random.randn(n_y, n_h) / np.sqrt(n_h) # initialization
	b = np.zeros((n_y,))

	#convert Y to Y_onehot
	Y_oh = convert_to_one_hot(Y, C=n_y)

	for t in range(num_iterations):
		for i in range(m):
			avg = sentence_to_avg(X[i], word_to_vec_map)

			z = np.dot(W, avg) + b
			a = softmax(z) # same shape with Y_oh

			cost = -np.sum(Y_oh[i]*np.log10(a))

			dz = a - Y_oh[i]
			dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
			db = dz

			W = W - learning_rate * dW
			b = b - learning_rate * db


		if t%100 == 0:
			print("Epoch: " + str(t) + " --- cost = " + str(cost))
			pred = predict(X, Y, W, b, word_to_vec_map)
	return pred, W, b

ped, W, b = model(X_train, Y_train, word_to_vec_map)
print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

########################## TEST PART #################################

#we can observe our model could detect some special words like "adore" and "love"
X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "not feeling happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4], [3]])
pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)
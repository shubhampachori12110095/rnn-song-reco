import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
from utils import *
import pickle

import matplotlib.pyplot as plt

catalog_size = 8000
unknown_token = "UNKNOWN_TOKEN"
seq_start_token = "SEQUENCE_START"
seq_end_token = "SEQUENCE_END"

print "Reading CSV file..."
with open('/Users/gaurav/Downloads/cleaned_full_result_36731371.csv', 'rU') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    sequences = itertools.chain(*[x[0].split(',') for x in reader])
    # Append SEQUENCE_START and SEQUENCE_END
    sequences = ["%s %s %s" % (seq_start_token, x, seq_end_token) for x in sequences]
print "Parsed %d sequences." % (len(sequences))

# Tokenize the sequences into songs
tokenized_sequences = [nltk.word_tokenize(seq) for seq in sequences]

# Count the word frequencies
song_freq = nltk.FreqDist(itertools.chain(*tokenized_sequences))
print "Found %d unique songs tokens." % len(song_freq.items())

# Get the most common songs and build index_to_song and song_to_index vectors
catalog = song_freq.most_common(catalog_size - 1)
index_to_song = [x[0] for x in catalog]
index_to_song.append(unknown_token)
song_to_index = dict([(w, i) for i, w in enumerate(index_to_song)])

print "Using catalog size %d." % catalog_size
print "The least frequent word in our catalog is '%s' and appeared %d times." % (catalog[-1][0], catalog[-1][1])

# Replace all songs not in our catalog with the unknown token
for i, seq in enumerate(tokenized_sequences):
    tokenized_sequences[i] = [w if w in song_to_index else unknown_token for w in seq]

# Create the training data
X_train = np.asarray([[song_to_index[w] for w in seq[:-1]] for seq in tokenized_sequences])
y_train = np.asarray([[song_to_index[w] for w in seq[1:]] for seq in tokenized_sequences])


class SongReco:
    def __init__(self, seq_dim, hidden_dim=100, bp_limit=4):
        self.seq_dim = seq_dim
        self.hidden_dim = hidden_dim
        self.bp_limit = bp_limit
        self.U = np.random.uniform(-np.sqrt(1. / seq_dim), np.sqrt(1. / seq_dim), (hidden_dim, seq_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (seq_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.seq_dim))
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)


np.random.seed(10)
model = SongReco(catalog_size)
o, s = model.forward_propagation(X_train[10])
print o.shape
print o

predictions = model.predict(X_train[10])
print predictions.shape
print predictions


def calculate_total_loss(self, x, y):
    L = 0
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        correct_song_predictions = o[np.arange(len(y[i])), y[i]]
        L += -1 * np.sum(np.log(correct_song_predictions))
    return L

def calculate_loss(self, x, y):
    N = np.sum((len(y_i) for y_i in y))
    return self.calculate_total_loss(x,y)/N

SongReco.calculate_total_loss = calculate_total_loss
SongReco.calculate_loss = calculate_loss

# Limit to 1000 examples to save time
print "Expected Loss for random predictions: %f" % np.log(catalog_size)
print "Actual loss: %f" % model.calculate_loss(X_train[:1000], y_train[:1000])

def bptt(self, x, y):
    T = len(y)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # Accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[np.arange(len(y)), y] -= 1.
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        # Backpropagation through time for bp_limit steps
        for bptt_step in np.arange(max(0, t-self.bp_limit), t+1)[::-1]:
            dLdW += np.outer(delta_t, s[bptt_step-1])
            dLdU[:,x[bptt_step]] += delta_t
            delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]

SongReco.bptt = bptt

np.random.seed(10)
model = SongReco(100, 10, bp_limit=1000)

# Performs one step of SGD.
def numpy_sdg_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW

SongReco.sgd_step = numpy_sdg_step

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


np.random.seed(10)
# Train on a small subset of the data to see what happens
model = SongReco(catalog_size)
losses = train_with_sgd(model, X_train[12000:15000], y_train[12000:15000], nepoch=10, evaluate_loss_after=1)

with open('hackmodel3_cleaned_mod.pkl', 'wb') as output:
    pickle.dump([model, catalog_size, unknown_token, seq_start_token, seq_end_token, catalog, index_to_song, song_to_index], output, pickle.HIGHEST_PROTOCOL)

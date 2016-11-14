from utils import *
import pickle
import json
import urllib2
from random import *
from types import *

class SongReco:
	def __init__(self, seq_dim, hidden_dim=100, bp_limit=4):
        self.seq_dim = seq_dim
        self.hidden_dim = hidden_dim
        self.bp_limit = bp_limit
        self.U = np.random.uniform(-np.sqrt(1. / seq_dim), np.sqrt(1. / seq_dim), (hidden_dim, seq_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (seq_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
	    T = len(x) #No. of songs in sequence
	    s = np.zeros((T + 1, self.hidden_dim)) #states
	    s[-1] = np.zeros(self.hidden_dim)
	    o = np.zeros((T, self.word_dim))
	    for t in np.arange(T):
	        s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
	        o[t] = softmax(self.V.dot(s[t]))
	    return [o, s]


with open('hackmodel3_cleaned.pkl', 'rb') as input:
    [model, catalog_size, unknown_token, seq_start_token, seq_end_token, catalog, index_to_song,
     song_to_index] = pickle.load(input)

    def generate_sequence(model, new_sequence):
        # Repeat until we get an end token
        i = 0
        while (new_sequence[-1] != song_to_index[seq_end_token] and i <= 10):
            next_song_probs = model.forward_propagation(new_sequence)
            sampled_word = song_to_index[unknown_token]
            # We don't want to sample unknown words
            samples = next_song_probs[0][-1]
            samples = np.argsort(samples)
            j=-1
            while sampled_word == song_to_index[unknown_token] or sampled_word in new_sequence or (sampled_word<=30 and new_sequence[-1]>=200):
                sampled_word = samples[j]
                j-=1

            new_sequence.append(sampled_word)
            i += 1
        sentence_str = [index_to_song[x] for x in new_sequence[0:-1]]
        return sentence_str

    #given a sequence of one or more songs, predict the next song or the complete sequence
    seq = [[song_to_index['3hI1iEK5']]]
    num_sequences = len(seq)
    for i in range(num_sequences):
        s = seq[i]
        s = generate_sequence(model, s)
        seq=''
        k=1
        for l in s:
            print l
            if l==seq_end_token:
                continue
            url = 'http://www.saavn.com/api.php?__call=song.getDetails&pids=' + l + '&_format=json&_marker=0&api_version=4'
            response = urllib2.urlopen(url)
            data = response.read()
            values = json.loads(data)
            seq+=str(k)+": "+values[l]["title"]+" ("+str(song_to_index[l])+")"+" | "
            k+=1
        print seq[:-3]
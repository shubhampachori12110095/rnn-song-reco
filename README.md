# rnn-song-reco
Recurrent neural network based real time song recommendation (Saavn)

Code to train a recurrent neural network on song sequence data. Song sequence data consists of sequences where each sequence is an ordered list of song ids played by a user in a session.

The format of input data is as follows - 
Each line in the text file is one sequence of song ids separated by comma.
Example - 
324321, 3241142, 5346543, 4235314, 32452345
34524325, 4365436, 3412311
5346543, 4235314, 32452345

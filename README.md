# Melody-generator
An attempt to make a melody generator using LSTM and RNN.
Basic idea here is to see the melody as a time-series representation, so now melody generation problem is equivalent to time-series prediction problem.
So we have to convert notes to a vocabulary of acceptable musical events like notes of different pitches and rests of given time interval.
Training:
We pass in chunks of melodies, then we ask lstm to predict next notes in the melody, lstm will be able to pick the pattern.
Melody generation:
We start with a seed melody: it is like a series of few notes that we feed into lstm then model will predict the next note.
Then we’ll append that to initial seed and feed that once again.
Yet another prediction which’ll give next note.
After doing that enough number of times, we get a whole melody.

Why rnn-lstm:
Melodies have long term structure patterns:
Patterns that repeat..
So even if we look at the most simple and beautiful tone which is twinkle twinkle little star, the phrases repeat itself.
So consider this section:
Twinkle twinkle little star, 
How i wonder what you are: this is a tone above the 1st part,
Up above the world so high: going up by a tone,
Like a diamond in the sky.
Lstms are good at capturing these long term temporal dependencies.

Dataset: 
ESAC dataset : Essen Associative Code and folksong database


Melody1 is one such melody generated using this model.

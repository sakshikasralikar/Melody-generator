import os
from typing import Mapping, Sequence
import music21 as m21
import json
import tensorflow.keras as keras
from music21 import *
import numpy as np
us = environment.UserSettings()


us.getSettingsPath()

m21.environment.set("musicxmlPath", r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe")

kern_dataset_path=r"essen\europa\deutschl\erk"
acceptable_durations=[0.25,0.5,0.75,1,1.5,2,3,4]
save_dir ="dataset"
single_file_dataset ="file_dataset"
sequence_length =64
mapping_path ="mapping.json"

def load_songs_in_kern(dataset_path):

    songs=[]
    #go through all the files in dataset and load them with music21
    for path,subdir,files in os.walk(dataset_path):
        for file in files:
            if file[-3:] =="krn":
                song= m21.converter.parse(os.path.join(path,file))
                songs.append(song)
                #song.show()

    return songs
def transpose(song):
    #get the key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measure_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key =measure_part0[0][4]

    #esimate key using musi21
    if not isinstance(key,m21.key.Key):
        key = song.analyze("key")
    #print(key)
    #get interval for transposition eg cmaj -> amin
    if key.mode =="major":
        interval =m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode =="minor":
        interval =m21.interval.Interval(key.tonic,m21.pitch.Pitch("A"))

    #transpose song by calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song

def encode_song(song,time_step=0.25):
    #encoding to time-series representation
    # p=60, d=1.0 ->[60,"-" "_", "_"]

    encoded_song=[]
    for event in song.flat.notesAndRests:
        # handling notes

        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi #midi

        #handle rests
        elif isinstance(event,m21.note.Rest):
            symbol ="r"
        #convert the note or rest into time series notation
        steps = int(event.duration.quarterLength/time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # cast encoded song to a str
    encoded_song =" ".join(map(str, encoded_song))

    return encoded_song

def has_acceptable_duration(song,acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def preprocess(dataset_path):
    pass
    #load the folk songs
    print("Loading songs...")
    songs=load_songs_in_kern(dataset_path)
    print(f"Loaded{len(songs)} songs.")


    #filter out songs that have non acceptable durations
    for i, song in enumerate(songs):
        if not has_acceptable_duration(song, acceptable_durations):
            continue

    
    #transpose songs to C major or A minor

        song=transpose(song)

    #encode the songs with music time series representation

        encoded_song = encode_song(song)

    #save songs to text file
        save_path =os.path.join(save_dir, str(i) )
        with open(save_path,"w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp:
        song =fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter="/ " * sequence_length
    songs = ""
    #load encoded songs and add delimiters
    for path,_, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path,file)
            song = load(file_path)
            songs = songs+song +" " +new_song_delimiter

    songs = songs[:-1]

    #save string that contains all dataset
    with open(file_dataset_path,"w") as fp:
        fp.write(songs)
    return songs

def create_mapping(songs, mapping_path):
    mappings = {}
    #identifying the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    #create mappings
    for i,symbol in enumerate(vocabulary):
        mappings[symbol]=i

    #save vocabulary to json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings,fp, indent=4)

def convert_songs_to_int(songs):
    int_songs=[]
    #load mappings
    with open(mapping_path, "r") as fp:
        mappings = json.load(fp)
    #cast songs string to a list
    songs =songs.split()
    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
    return int_songs


def generate_training_sequences(sequence_length):

    #load the songs and map them to int
    songs =load(single_file_dataset)
    int_songs=convert_songs_to_int(songs)

    #generate the training seq
    #100 symbols , 64 sl, 100-64
    inputs=[]
    targets=[]

    num_sequences=len(int_songs)-sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    #one-hot encode the sequences
    #inputs: #of seq, sl, vocab size
    #{[]}
    vocabulary_size =len(set(int_songs))
    inputs= keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets =np.array(targets)

    return inputs, targets

def main():

    preprocess(kern_dataset_path)
    songs = create_single_file_dataset(save_dir,single_file_dataset,sequence_length)
    create_mapping(songs, mapping_path)
    inputs,targets = generate_training_sequences(sequence_length)
    
    
if __name__ == '__main__':
    main()

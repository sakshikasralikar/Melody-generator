import tensorflow.keras as keras
import json
from preprocess import sequence_length, mapping_path
import numpy as np
import music21 as m21

from pygame import mixer

class MelodyGenerator:
    def __init__(self,model_path="./model.h5"):

        self.model_path=model_path
        self.model=keras.models.load_model(model_path)

        with open(mapping_path,"r") as fp:
            self._mappings = json.load(fp)
        
        self._start_symbols =["/"] * sequence_length

    def generate_melody(self,seed,num_steps, max_sequence_length, temperature):

        #create seed with start symbol
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            #limit the seed to max_seq_llength
            seed = seed[-max_sequence_length:]

            #one hod encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            #(1,max-seq_length, num of symbols in vocab)
            onehot_seed =onehot_seed[np.newaxis, ...]

            #make predictions
            probabilities = self.model.predict(onehot_seed)[0]
            output_int =self._sample_with_temperature(probabilities,temperature)

            #update seed
            seed.append(output_int)

            #map int to our encoding
            output_symbol =[k for k,v in self._mappings.items() if v==output_int][0]

            #check whether we are at the end of the melody

            if output_symbol =="/":
                break
            #update the melody
            melody.append(output_symbol)
        
        return melody

        

    def _sample_with_temperature(self, probabilities, temperature):
        predictions =np.log(probabilities)/temperature
        probabilities=np.exp(predictions)/np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index =np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, format ="midi",step_duration=0.25, file_name="mel.mid"):
        #create a music21 stream
        stream =m21.stream.Stream()
        #parse all the symbols in the melody and creae note/rests objects
        #60 _ _ _ r _ 62 _
        start_symbol =None
        step_counter = 1

        for i, symbol in enumerate(melody):
            #handle case in which we have to a note/rest
            if symbol!="_" or i+1==len(melody):
                # ensure we're dealing with note/rest boyond the first one
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter
                    #handle rest
                    if start_symbol =="r":
                        m21_event = m21.note.Rest(quarterLength =quarter_length_duration)

                    #handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol),quarterLength = quarter_length_duration )

                    stream.append(m21_event)

                    step_counter =1
                start_symbol=symbol
                
            #handle case in which we have a prolongation sign "_"
            else: 
                step_counter +=1
        #write the m21 stream to a midi file
        stream.write(format, file_name)

if __name__ == '__main__':
    mg=MelodyGenerator()
    seed ="55 _ _ _ 60 _ 64 _ 67 _ _ _ 64 "
    seed2 = "55 _ _ _ 60 _ _ _ 60 _ _ _ 55 _ _ _ 55"
    
    melody =mg.generate_melody(seed2,500, sequence_length, 0.1)
    print(melody)
    mg.save_melody(melody)
    mixer.init()
  
    # Loading the song
    mixer.music.load("mel.mid")
    
    # Setting the volume
    mixer.music.set_volume(0.7)
    
    # Start playing the song
    mixer.music.play()
    while True:
      
        print("Press 'p' to pause, 'r' to resume")
        print("Press 'e' to exit the program")
        query = input(" ")
        
        if query == 'p':
    
            # Pausing the music
            mixer.music.pause()     
        elif query == 'r':
    
            # Resuming the music
            mixer.music.unpause()
        elif query == 'e':
    
            # Stop the mixer
            mixer.music.stop()
            break

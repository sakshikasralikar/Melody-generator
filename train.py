from preprocess import generate_training_sequences,sequence_length
import tensorflow.keras as keras

Output_units = 38
Loss ="sparse_categorical_crossentropy"
Learning_rate=0.001
Num_units= [256]
Epochs = 60
Batch_size =64
save_model_path = "model.h5"
def build_model(output_units,num_units, loss, learning_rate):

    #create architecture
    input=keras.layers.Input(shape=(None, output_units))
    x=keras.layers.LSTM(num_units[0])(input)
    x=keras.layers.Dropout(0.2)(x)


    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input,output)

    #compile model
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(lr=learning_rate),
    metrics=["accuracy"])

    model.summary()

    return model


def train(output_units=Output_units,num_units=Num_units, loss=Loss, learning_rate=Learning_rate):

    #generate the training sequences
    inputs,targets = generate_training_sequences(sequence_length)
    #buid the network
    model = build_model(output_units,num_units, loss, learning_rate)
    #train the model
    model.fit(inputs, targets, epochs=Epochs, batch_size=Batch_size)
    #save the model
    model.save(save_model_path)

if __name__ == '__main__':
    train()
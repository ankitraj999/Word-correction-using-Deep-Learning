#importing all libraries
import numpy as np
import data_config
import data_generator
import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed, Bidirectional, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


#Preparing the dataset..if news.2013.en.shuffled is not present in data/processed folder it will create the news.2013.en.shuffled file containing processed text
saved_processed_data_path=os.path.join(data_config.processed_data,'news.2013.en.shuffled')
if not os.path.exists(saved_processed_data_path):
    data_generator.data_generator(data_config.unprocessed_file_names)


#Changing each word in text file into sequence of numbers 
def text_to_sequence(lines):
    sequences = []
    for i,line in enumerate(lines):
         if i%10000 == 0:
                sequence = [data_config.char_to_index[char] for char in line]

         sequences.append(sequence)
    return sequences

# padding the word to make constant word length of maxlen
def sequence_encoder(length, lines):
    X = text_to_sequence(lines)
    X = pad_sequences(X, maxlen=length, padding='post')
    return X

def output_encoder(sequences, vocab_size):
    ylist = list()
    for i, sequence in enumerate(sequences):

            encoded = to_categorical(sequence, num_classes=vocab_size)
            ylist.append(encoded)
    
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

#Model defination
def create_model(vocab_size = len(data_config.allowed_chars), source_timesteps = data_config.wrong_sent_len, target_timesteps = data_config.max_len_text, n_units = 250):
    model = Sequential()
    model.add(Embedding(vocab_size, n_units, input_length=source_timesteps, mask_zero=True))
    model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(n_units))
    model.add(Dropout(0.2))
    model.add(RepeatVector(target_timesteps))
    model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print(model.summary())
    return model

#Training the model with google's 1 billion words corpus with 10 epoch
def train_model(n_epochs=10, data_load_limit=100000):
    model = create_model()
    for _ in range(n_epochs):
        for file_name in data_config.processed_file_names:
            print("Reading:", file_name)
            current_batch = 0
            while True:
                lines = open(file_name).readlines()[current_batch*data_load_limit:][:data_load_limit]
                current_batch = current_batch + 1
                if not lines:
                    break
                X = []
                Y = []
                for i,line in enumerate(lines):
                    line = line.strip().lstrip()
                    line = line.split('\t')
                    if len(line) == 2:
                        X.append(line[0])
                        Y.append(line[1])
                    else:
                        print("Length less than 1:",line)
                    lines[i] = 0
                print(len(X))
                print(len(Y))
                X = sequence_encoder(data_config.wrong_sent_len, X)
                print(len(X))
                Y = sequence_encoder(data_config.max_len_text, Y)
                print(len(Y))
                Y = output_encoder(Y, len(data_config.allowed_chars))
                
                checkpoint = ModelCheckpoint('checkpoint_file', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
                print("Starting training for batch", current_batch, "of size", data_load_limit, "from file", file_name)
                model.fit(X, Y, epochs=1, batch_size=64, validation_split=0.01, callbacks=[checkpoint], verbose=1)
    
#Function to convert output of the model into text;
def convert_to_final_text(outcome):
    voc_class=outcome.shape[0]*outcome.shape[2]
    outcome=outcome.reshape(outcome.shape[1],voc_class)
    seq_with_pad=np.argmax(outcome,axis=1)
    aa=np.array([0])
    seq_without_pad=np.setdiff1d(seq_with_pad,aa)
    seq_without_pad=list(seq_without_pad)
    output_text = [data_config.index_to_char[i] for i in seq_without_pad]
    output_text=''.join(output_text)
    print("output: ")
    print(output_text)

#Main function
if __name__ == '__main__':
    #If checkpoit_file is not present it will train the model to create checkpoint_file otherwise existing checkpoint_file will be used to predict the incorrect words
    saved_model_path=os.path.join(data_config.project_path,'checkpoint_file')
    if not os.path.exists(saved_model_path):
        train_model()

    else: 
        model=load_model('checkpoint_file')
        #incorrect word can be used in Test_variable like
        input_list=list()
        print("Input")
        text_input=raw_input()
        input_list.append(text_input)
        out=text_to_sequence(input_list)
        X = pad_sequences(out, maxlen=data_config.wrong_sent_len, padding='post')
        outcome=model.predict(X,verbose=0)
        #convert to predicted text
        convert_to_final_text(outcome)



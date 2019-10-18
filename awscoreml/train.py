import os
import re
import json
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
import pickle


from awscoreml.resolve import paths


def read_config_file(config_json):
    """
    This function reads in a json file like hyperparameters.json or resourceconfig.json
    :param config_json: this is a string path to the location of the file (for both sagemaker or local)
    :return: a python dict is returned
    """
    config_path = paths.config(config_json)
    if os.path.exists(config_path):
        json_data = open(config_path).read()
        return json.loads(json_data)


def preprocess_tweet(tweet):
    """
    preprocess the text in a single tweet. convert all urls to sting "URL"
    after that convert all @username to "AT_USER" then correct all multiple white spaces to a single white space
    finally convert "#topic" to just "topic".
    :param tweet:
    :return: the clean version of that single tweet
    """
    tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = re.sub(r'\W*\b\w{1,3}\b', '', tweet)
    return tweet


def entry_point():
    """
    This function trains the model prameters and same them
    read data , describe model graph and finally train model
    return: initiates the keras training job and saved model.h5 file at the end
    """
    dataframe = pd.read_csv(paths.input(channel='validation', filename="training.1600000.processed.noemoticon.csv"), error_bad_lines=False, encoding="ISO-8859-1", header=None).iloc[:, [0, 4, 5]].sample(frac=1).reset_index(drop=True)
    tweets = np.array(dataframe.iloc[:, 2].apply(preprocess_tweet).values)
    sentiment = np.array(dataframe.iloc[:, 0].values)

    hyper_params = read_config_file('hyperparameters.json')
    vocab_size = int(30)

    tk = Tokenizer(num_words=vocab_size)
    tk.fit_on_texts(tweets)

    with open(paths.model('tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    t = tk.texts_to_sequences(tweets)
    X = np.array(sequence.pad_sequences(t, maxlen=20, padding='post'))
    y = sentiment
    print(X.shape, y.shape)
    y[y == 4] = 1

    model = Sequential()
    model.add(Embedding(vocab_size, 32, input_length=20))
    model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=7, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=8, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.summary()

    history = model.fit(
        X, y,
        batch_size=int(10),
        verbose=1,
        validation_split=float(0.2),
        epochs=int(1)
    )

    model.save(paths.model(filename='model.h5'))
'''
    print("loss:" + str(history.history['loss']))
    print("acc:" + str(history.history['acc']))
    print("val_loss" + str(history.history['val_loss']))
    print("val_acc" + str(history.history['val_acc']))

'''
if __name__ == '__main__':
    entry_point()

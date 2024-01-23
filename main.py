import tensorflow as tf
import os
from prettytable import PrettyTable
import numpy as np
import pickle
from keras.layers import Dense, LSTM, Embedding, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

maxWordsCount = 20000
max_text_len = 100
tokenizer = Tokenizer()
model = Model()

print("Выберите режим")
print("0 - без обучения, используя предварительные веса и модель")
print("1 - с обучением (очень долго, много данных для обучения)")
version = int(input())

if version == 0:
    model = tf.keras.models.load_model('my_model.h5')
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

if version == 1:
    tokenizer = Tokenizer(filters='!–"—#$%&amp;,()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                          lower=True, split=' ', char_level=False)

    with open('train_data_true.txt', 'r', encoding='utf-8') as f:
        texts_true = f.readlines()
        texts_true[0] = texts_true[0].replace('\ufeff', '')
    f.close()
    with open('train_data_main.txt', 'r', encoding='utf-8') as f:
        texts_main = f.readlines()
        texts_main[0] = texts_main[0].replace('\ufeff', '')
    f.close()
    with open('train_data_false.txt', 'r', encoding='utf-8') as f:
        texts_false = f.readlines()
        texts_false[0] = texts_false[0].replace('\ufeff', '')
    f.close()

    all_texts = texts_main + texts_true + texts_false
    tokenizer.fit_on_texts(all_texts)

    data_main = tokenizer.texts_to_sequences(texts_main)
    data_true = tokenizer.texts_to_sequences(texts_true)
    data_false = tokenizer.texts_to_sequences(texts_false)

    data_pad_main = pad_sequences(data_main, maxlen=max_text_len)
    data_pad_true = pad_sequences(data_true, maxlen=max_text_len)
    data_pad_false = pad_sequences(data_false, maxlen=max_text_len)

    X = []
    Y = []

    for main_row, main_row2, true_row, false_row in zip(data_pad_main, data_pad_main, data_pad_true, data_pad_false):
        X.append([main_row, true_row])
        Y.append([1, 0])

        X.append([main_row2, false_row])
        Y.append([0, 1])

    X = np.array(X)
    Y = np.array(Y)

    input_main_text = Input(shape=(max_text_len,))
    input_related_text = Input(shape=(max_text_len,))
    embedding_layer = Embedding(maxWordsCount, 128, input_length=max_text_len)
    embedded_main_text = embedding_layer(input_main_text)
    embedded_related_text = embedding_layer(input_related_text)

    lstm_layer = LSTM(128, activation='tanh', return_sequences=True)
    lstm_output_main_text = lstm_layer(embedded_main_text)
    lstm_output_related_text = lstm_layer(embedded_related_text)
    concatenated_output = concatenate([lstm_output_main_text, lstm_output_related_text])

    dense_layer = LSTM(64, activation='tanh')(concatenated_output)
    output = Dense(2, activation='sigmoid', name='output')(dense_layer)
    model = Model(inputs=[input_main_text, input_related_text], outputs=output)
    model.summary()
    #optimizer попробовать sgd
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

    history = model.fit([X[:, 0], X[:, 1]], Y, batch_size=32, epochs=15)

    model.save('my_model.h5')
    with open('tokenizer.pkl', 'wb') as file:
        pickle.dump(tokenizer, file)

os.chdir("Education_list")
files = os.listdir()
col = len(files) + 1
print(files)
a = [[""] * col for i in range(col)]
for n, item1 in enumerate(files):
    with open(item1, 'r', encoding='utf-8') as f:
        main_text = f.readline()
        new_main_text = f.readline()
        a[n + 1][0] = main_text
    for k, item in enumerate(files):
        if k < n:
            with open(item, 'r', encoding='utf-8') as f1:
                related_text = f1.readline()
                new_related_text = f1.readline()
                a[0][k + 1] = related_text

            two_texts = new_main_text + new_related_text
            tokenizer.fit_on_texts(two_texts)

            new_main_sequence = tokenizer.texts_to_sequences([new_main_text])
            new_related_sequence = tokenizer.texts_to_sequences([new_related_text])

            new_padded_main_sequence = pad_sequences(new_main_sequence, maxlen=max_text_len)
            new_padded_related_sequence = pad_sequences(new_related_sequence, maxlen=max_text_len)

            new_X = [new_padded_main_sequence, new_padded_related_sequence]
            prediction = model.predict(new_X)

            predicted_label = "+" if prediction[0][0] >= 0.5 else "-"
            a[n + 1][k + 1] = predicted_label
            a[k + 1][n + 1] = predicted_label
            print("", main_text, related_text, predicted_label, prediction[0][0])

            f.close()
            f1.close()
        else:
            break
a[0][col - 1] = a[col - 1][0]

table = PrettyTable()
for row in a:
    table.add_row(row, divider=True)
table.header = False
print(table)

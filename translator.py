import io
import os.path
import string
import sys
import pickle
import zipfile

import keras.losses
import pandas as pd
import unidecode
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, LSTM, RepeatVector, Dense
from keras.models import load_model
from keras.optimizer_v2.rmsprop import RMSprop
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from numpy import array
from sklearn.model_selection import train_test_split


OUTPUT_DIM = 1024


def prepare_text_zip(input_archive:str, input_file:str, output_dir:str):
    """

    :param input_archive: path to zip file containing the input_file
    :param input_file: name of file inside input_archive containing raw training data
    :param output_file: path for saving pickle list containing processed training data
    """

    data = []
    trans_punctuation = str.maketrans('', '', string.punctuation)

    with zipfile.ZipFile(input_archive) as zfile:
        with zfile.open(input_file) as readfile:
            for line in io.TextIOWrapper(readfile, 'utf-8'):
                line_str = str(line)
                clean_str = unidecode.unidecode(line_str)\
                    .strip()\
                    .lower()\
                    .translate(trans_punctuation)

                data += [clean_str.split("\t")]

    lines_array = array(data)

    # Limit lines to fit in GPU memory
    lines_array = lines_array[:2500, :]

    # Remove attribution field
    lines_array = lines_array[:, :2]

    print(lines_array)

    save_input_data(lines_array, output_dir)


def prepare_text(input_file:str, output_dir:str):
    """
    :param input_file: path to text file containing raw training data
    :param output_file: path for saving pickle list containing processed training data
    """

    with open(input_file, mode='rt', encoding='utf-8') as file_in:
        data = file_in.read()

    # TODO:
    # lang = lang_file[:3] ???

    # Remove accents
    data = unidecode.unidecode(data)

    # Remove extra whitespace
    data = data.strip()

    # Make lowercase
    data = data.lower()

    # Remove punctuation
    trans_punctuation = str.maketrans('', '', string.punctuation)
    data = data.translate(trans_punctuation)

    # split into lines
    lines = data.split('\n')

    # split into sentences in,out
    lines = [i.split('\t') for i in lines]

    lines_array = array(lines)

    # Limit lines to fit in GPU memory
    lines_array = lines_array[:2500, :]

    # Remove attribution field
    lines_array = lines_array[:, :2]

    print(lines_array)

    save_input_data(lines_array, output_dir)


def save_input_data(lines_array, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # save pickle (used for creating translator)
    output_file = os.path.join(output_dir, "phrases.pic")
    with open(output_file, "wb") as file_out:  # Pickling
        pickle.dump(lines_array, file_out)
    # save plain text (for debug)
    output_file = os.path.join(output_dir, "phrases.txt")
    with open(output_file, "wt") as text_out:
        for line in lines_array:
            text_out.write("%s\t%s\n" % (line[0], line[1]))


# def init_cuda():
#     from tensorflow.compat.v1.keras.backend import set_session
#     import tensorflow as tf
#     config = tf.compat.v1.ConfigProto()
#     config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#     config.gpu_options.experimental.use_cuda_malloc_async = True #This leverages the new async allocation APIs.
#     config.log_device_placement = True  # to log device placement (on which device the operation ran)
#     sess = tf.compat.v1.Session(config=config)
#     set_session(sess)


def create_translator(input_dir: str, translator_dir: str):
    # init_cuda()

    input_file = os.path.join(input_dir, "phrases.pic")
    with open(input_file, "rb") as file_in:  # Unpickling
        sentences = pickle.load(file_in)
        print(sentences)

    # function to build a tokenizer
    def tokenization(lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    # prepare data inputs
    tokenizer_in = tokenization(sentences[:, 0])
    tokenizer_in_length = len(tokenizer_in.word_index) + 1

    tokenizer_out = tokenization(sentences[:, 1])
    tokenizer_out_length = len(tokenizer_out.word_index) + 1

    os.makedirs(translator_dir, exist_ok=True)

    translator_file = os.path.join(translator_dir, "translator.h5")
    checkpoint = ModelCheckpoint(translator_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    print('Input Vocabulary Size: %d' % tokenizer_in_length)
    print('Output Vocabulary Size: %d' % tokenizer_out_length)

    # encode and pad sequences
    def encode_sequences(tokenizer, length, lines):
        # integer encode sequences
        seq = tokenizer.texts_to_sequences(lines)
        # pad sequences with 0 values
        seq = pad_sequences(seq, maxlen=length, padding='post')
        return seq

    # split data into train and test set
    train, test = train_test_split(sentences, test_size=0.2, random_state=12)

    # prepare training data
    trainX = encode_sequences(tokenizer_out, tokenizer_out_length, train[:, 1])
    trainY = encode_sequences(tokenizer_in, tokenizer_in_length, train[:, 0])

    # prepare and save validation data
    testX = encode_sequences(tokenizer_out, tokenizer_out_length, test[:, 1])
    testY = encode_sequences(tokenizer_in, tokenizer_in_length, test[:, 0])

    pickle.dump(test, open(os.path.join(translator_dir, "test.pkl"), 'wb'))
    pickle.dump(testX, open(os.path.join(translator_dir, "testX.pkl"), 'wb'))

    # build NMT model
    def define_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
        new_model = Sequential()
        new_model.add(Embedding(input_dim=in_vocab, output_dim=units, input_length=in_timesteps, mask_zero=True))
        new_model.add(LSTM(units=units))
        new_model.add(RepeatVector(out_timesteps))
        new_model.add(LSTM(units=units, return_sequences=True))
        new_model.add(Dense(units=out_vocab, activation='softmax'))
        return new_model

    in_length = 8 # tokenizer_out_length
    out_length = 8 # tokenizer_in_length
    model = define_model(tokenizer_in_length, tokenizer_out_length, in_length, out_length, OUTPUT_DIM)

    rms = RMSprop(learning_rate=0.001)
    model.compile(optimizer=rms, loss=keras.losses.sparse_categorical_crossentropy) # loss='sparse_categorical_crossentropy')

    model.summary()

    def plot(model):
        from keras.utils.vis_utils import plot_model
        plot_model(model, show_shapes=True, to_file=os.path.join(translator_dir, "translator.png"))

    plot(model)

    # train model
    history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                        epochs=30, batch_size=512, validation_split = 0.2, callbacks=[checkpoint],
                        verbose=1)

    pickle.dump(history, open(os.path.join(translator_dir, "history.pkl"), 'wb'))



def run_translator(translator_dir: str, input_dir: str):
    """

    :param translator_file: trained NN to load into translator
    :param input_file: path to text file containing lines of text to be translated
    :param output_file: path to where translated lines will be written
    """

    translator_file = os.path.join(translator_dir, "translator.h5")
    model = load_model(translator_file)

    test = pickle.load(open(os.path.join(translator_dir, "test.pkl"), 'rb'))
    testX = pickle.load(open(os.path.join(translator_dir, "testX.pkl"), 'rb'))

    testX.reshape((testX.shape[0], testX.shape[1]))
    preds = model.predict(testX)


    y_pred = preds[5]
    preds_nn = []

    for k in range(8):
        words_idx = [i for i, prob in enumerate(y_pred[k]) if prob > 0.5]
        preds_nn = preds_nn + words_idx


    all_preds_nn = []
    it = 0
    for y_pred in preds[range(1000)]:
        preds_nn = []
        for k in range(8):
            words_idx = [i for i, prob in enumerate(y_pred[k]) if prob > 0.5]
            preds_nn = preds_nn + words_idx

        if it % 100 == 0:
            print(it)

        all_preds_nn.append(preds_nn)
        it = it + 1

    def get_word(n, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == n:
                return word
        return None

    def tokenization(lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    input_file = os.path.join(input_dir, "phrases.pic")
    with open(input_file, "rb") as file_in:  # Unpickling
        sentences = pickle.load(file_in)

    eng_tokenizer = tokenization(sentences[:, 0])


    preds_text = []
    for i in all_preds_nn:
        temp = []
        for j in range(len(i)):
            t = get_word(i[j], eng_tokenizer)
            if j > 0:
                if (t == get_word(i[j - 1], eng_tokenizer)) or (t == None):
                    temp.append('')
                else:
                    temp.append(t)
            else:
                if (t == None):
                    temp.append('')
                else:
                    temp.append(t)

        preds_text.append(' '.join(temp))

    pred_df = pd.DataFrame({'actual' : test[range(1000),0], 'predicted' : preds_text})

    pred_df.sample(250)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        command = sys.argv[1]
        if command == "prepare":
            if len(sys.argv) == 5:
                input_archive = sys.argv[2]
                input_file = sys.argv[3]
                output_dir = sys.argv[4]
                print("Prepare data: %s::%s -> %s" % (input_archive, input_file, output_dir))
                prepare_text_zip(input_archive, input_file, output_dir)
            else:
                input_file = sys.argv[2]
                output_dir = sys.argv[3]
                print("Prepare data: %s -> %s" % (input_file, output_dir))
                prepare_text(input_file, output_dir)

        if command == "create":
            input_dir = sys.argv[2]
            translator_dir = sys.argv[3]
            print("Create translator : %s -> %s" % (input_dir, translator_dir))
            create_translator(input_dir, translator_dir)

        if command == "run":
            input_dir = sys.argv[2]
            translator_dir = sys.argv[3]
            print("Run translator : %s, %s" % (input_dir, translator_dir))
            run_translator(input_dir, translator_dir)

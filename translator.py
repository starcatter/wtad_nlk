import io
import logging
import os
import pickle
import sys
import zipfile
from datetime import datetime
from argparse import ArgumentParser

##############################
# set TF logging level BEFORE importing TF dependency
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras_nlp
import unidecode
from keras.api.keras import layers
from keras.layers import TextVectorization

import random
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils.vis_utils import plot_model


def prepare_text_zip(input_archive: str,
                     input_file: str,
                     output_dir: str,
                     flip_translation=False,
                     limit_lines=50000,
                     vocab_size=50000,
                     sequence_length=16):
    """
    :param input_archive: path to zip file containing the input_file
    :param input_file: name of file inside input_archive containing raw training data
    :param output_dir: path for saving pickle list containing processed training data
    :param flip_translation: select text data for eng->target or target->eng translation
    :param limit_lines: max number of lines to train
    :param vocab_size: vectorizer size
    :param sequence_length: max sequence length in tokens/words
    """
    # generate a random seed for working with this dataset
    random.seed(random.choice([0xDEADBEEF, 0xBADF00D, 0xDECAFBAD, 0x1BADB002]))
    pickled_seed = int(random.random()*1_000_000)

    text_pairs = []
    trans_punctuation = str.maketrans('', '', string.punctuation)

    with zipfile.ZipFile(input_archive) as zfile:
        with zfile.open(input_file) as readfile:
            for line in io.TextIOWrapper(readfile, 'utf-8'):
                line_str = str(line)
                clean_str = unidecode.unidecode(line_str) \
                    .strip() \
                    .lower() \
                    .translate(trans_punctuation)

                eng, target, _ = clean_str.split("\t")

                # 1. handle bi-directional translation
                # 2. wrap target sentence in [start]/[stop] tags
                if flip_translation:
                    text_pairs.append((target, "[start] " + eng + " [end]"))
                else:
                    text_pairs.append((eng, "[start] " + target + " [end]"))

    random.shuffle(text_pairs)

    # Limit lines to fit in GPU memory
    if len(text_pairs) > limit_lines:
        text_pairs = text_pairs[:limit_lines]

    # Print a few pairs
    for _ in range(5):
        print(random.choice(text_pairs))

    # split the sentence pairs into a training set, a validation set, and a test set.
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples: num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples:]
    test_texts = [pair[0] for pair in test_pairs]

    print(f"{len(text_pairs)} total pairs")
    print(f"{len(train_pairs)} training pairs")
    print(f"{len(val_pairs)} validation pairs")
    print(f"{len(test_pairs)} test pairs")

    source_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize=None
    )

    target_vectorization = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=None
    )

    train_source_texts = [pair[0] for pair in train_pairs]
    train_target_texts = [pair[1] for pair in train_pairs]

    # adapt vectorization
    source_vectorization.adapt(train_source_texts)
    target_vectorization.adapt(train_target_texts)

    os.makedirs(output_dir, exist_ok=True)

    # save pickle (used for creating translator)
    pickled_data = {
        "source_vectorization": {
            'config': source_vectorization.get_config(),
            'weights': source_vectorization.get_weights()
        },
        "target_vectorization": {
            'config': target_vectorization.get_config(),
            'weights': target_vectorization.get_weights()
        },
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "test_pairs": test_pairs,

        "vocab_size": vocab_size,
        "sequence_length": sequence_length,

        "seed": pickled_seed
    }

    output_file = os.path.join(output_dir, "text_data.pic")
    with open(output_file, "wb") as file_out:
        pickle.dump(pickled_data, file_out)

    # save plain texts (for debug)
    for text_file, textSource in [("train_pairs.txt", train_pairs), ("val_pairs.txt", val_pairs),
                                  ("test_pairs.txt", test_pairs)]:
        output_file = os.path.join(output_dir, text_file)
        with open(output_file, "wt") as text_out:
            for line in textSource:
                text_out.write("%s\t%s\n" % (line[0], line[1].replace("[start]", "").replace("[end]", "")))

    # save test texts for easy testing
    output_file = os.path.join(output_dir, "test_texts.txt")
    with open(output_file, "wt") as text_out:
        for line in test_texts:
            text_out.write(line + "\n")


def load_text_data(data_dir):
    """
    Loads pickled text data and reinits TextVectorization
    :param data_dir:
    :return:
    """
    input_file = os.path.join(data_dir, "text_data.pic")
    with open(input_file, "rb") as file_in:
        pickled_data = pickle.load(file_in)

    source_vectorization = TextVectorization.from_config(pickled_data['source_vectorization']['config'])
    target_vectorization = TextVectorization.from_config(pickled_data['target_vectorization']['config'])

    # You have to call `adapt` with some dummy data (BUG in Keras)
    source_vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    source_vectorization.set_weights(pickled_data['source_vectorization']['weights'])

    target_vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    target_vectorization.set_weights(pickled_data['target_vectorization']['weights'])

    return pickled_data, source_vectorization, target_vectorization


def create_datasets(pickled_data, source_vectorization, target_vectorization, batch_size):
    """
    Helper for creating training datasets
    based on https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
    :param pickled_data:
    :param source_vectorization:
    :param target_vectorization:
    :param batch_size:
    :return:
    """
    def format_dataset(source, target):
        source = source_vectorization(source)
        target = target_vectorization(target)
        return (
            {
                "encoder_inputs": source,
                "decoder_inputs": target[:, :-1],
            },
            target[:, 1:],
        )

    def make_dataset(pairs):
        source_texts, target_texts = zip(*pairs)
        source_texts = list(source_texts)
        target_texts = list(target_texts)
        dataset = tf.data.Dataset.from_tensor_slices((source_texts, target_texts))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(format_dataset)
        return dataset.shuffle(2048, seed=pickled_data["seed"]).prefetch(16).cache()

    train_ds = make_dataset(pickled_data["train_pairs"])
    val_ds = make_dataset(pickled_data["val_pairs"])

    return train_ds, val_ds


class TransformerDecoder(layers.Layer):
    """
    TransformerDecoder implementation
    keras_nlp.layers.TransformerEncoder performed significantly worse
    Based on https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
    """
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True
        self.padding_mask = None

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            self.padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            self.padding_mask = tf.minimum(self.padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=self.padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update({"embed_dim": self.embed_dim})
        config.update({"latent_dim": self.latent_dim})
        config.update({"num_heads": self.num_heads})
        return config

    # There's actually no need to define `from_config` here, since returning
    # `cls(**config)` is the default behavior.
    @classmethod
    def from_config(cls, config):
        # self.embed_dim = config["embed_dim"]
        # self.latent_dim = config["latent_dim"]
        # self.num_heads = config["num_heads"]
        return cls(**config)


def create_translator(input_dir: str, translator_dir: str, batch_size=256, embed_dim=256, latent_dim=1024, num_heads=6,
                      dropout=0.0125, epochs=4):
    """
    Builds a end-to-end translation model

    :param input_dir:
    :param translator_dir:
    :param batch_size:
    :param embed_dim:
    :param latent_dim:
    :param num_heads:
    :param dropout:
    :param epochs:
    """
    pickled_data, source_vectorization, target_vectorization = load_text_data(input_dir)
    train_ds, val_ds = create_datasets(pickled_data, source_vectorization, target_vectorization, batch_size)

    for inputs, targets in train_ds.take(1):
        print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
        print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
        print(f"targets.shape: {targets.shape}")

    # assemble the end-to-end model.

    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")

    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=pickled_data["vocab_size"],
        sequence_length=pickled_data["sequence_length"],
        embedding_dim=embed_dim
    )(encoder_inputs)

    encoder_transformer = keras_nlp.layers.TransformerEncoder(intermediate_dim=latent_dim,
                                                              num_heads=num_heads,
                                                              name="TransformerEncoder",
                                                              dropout=dropout)
    encoder_outputs = encoder_transformer(x)

    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")

    decoder_embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(pickled_data["vocab_size"],
                                                                         pickled_data["sequence_length"],
                                                                         embed_dim
                                                                         )
    x = decoder_embedding_layer(decoder_inputs)

    # using custom class
    x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)

    x = layers.Dropout(dropout)(x)

    decoder_outputs = layers.Dense(pickled_data["vocab_size"], activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )

    transformer.summary()
    transformer.compile(
        "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    os.makedirs(translator_dir, exist_ok=True)
    try:
        plot_model(transformer, show_shapes=True, to_file=os.path.join(translator_dir, "transformer.png"))
    except ImportError:
        pass

    # must train at least one epoch before saving
    transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)

    transformer.save(os.path.join(translator_dir, "transformer.ts"))


def train_model(input_dir: str, translator_dir: str, epochs=16):
    """
    Allows training loaded models. Will overwrite the previous model.
    :param input_dir:
    :param translator_dir:
    :param epochs:
    """
    pickled_data, source_vectorization, target_vectorization = load_text_data(input_dir)
    train_ds, val_ds = create_datasets(pickled_data, source_vectorization, target_vectorization, 256)
    model = keras.models.load_model(os.path.join(translator_dir, "transformer.ts"))
    model.fit(train_ds, epochs=epochs, validation_data=val_ds)
    model.save(os.path.join(translator_dir, "transformer.ts"))


def translate_sequences(input_dir: str, translator_dir: str, sequences:[str]):
    """
    Uses a trained model to translate string sequence list passed via @sequences param
    :param input_dir:
    :param translator_dir:
    :param sequences:
    """
    pickled_data, source_vectorization, target_vectorization = load_text_data(input_dir)

    target_vocab = target_vectorization.get_vocabulary()
    target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))

    model = keras.models.load_model(os.path.join(translator_dir, "transformer.ts"))

    results = []
    for input_sentence in sequences:
        input_sentence_clean = input_sentence.strip().replace("\n", "")
        tokenized_input_sentence = source_vectorization([input_sentence_clean])
        decoded_sentence = "[start]"
        for i in range(pickled_data["sequence_length"]):
            tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
            predictions = model([tokenized_input_sentence, tokenized_target_sentence])

            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = target_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == "[end]":
                break

        results.append( (input_sentence_clean, decoded_sentence.replace("[start]", "").replace("[end]", "") ))

    return results


"""
Command line handlers
"""

def handle_prep(args):
    prepare_text_zip(args.archive,
                     args.file,
                     args.dir,
                     args.flip,
                     args.limit_lines,
                     args.vocab_size,
                     args.sequence_length)


def handle_create(args):
    create_translator(args.lang, args.model,
                      args.batch_size,
                      args.embed_dim,
                      args.latent_dim,
                      args.num_heads,
                      args.dropout,
                      args.epochs)

def handle_train(args):
    train_model(args.lang,
                args.model,
                args.epochs)


def handle_translate(args):
    if args.phrase:
        for seq_in,seq_out in translate_sequences(args.lang, args.model, [args.phrase]):
            print("in:[%s] -> out:[%s]" % (seq_in, seq_out))
    elif args.file:
        with open(args.file, "rt") as file_in:
            lines = file_in.readlines()

        if args.sample:
            lines = random.sample(lines, args.sample )

        for seq_in,seq_out in translate_sequences(args.lang, args.model, lines):
            print("in:[%s] -> out:[%s]" % (seq_in, seq_out))

"""
Command line parser
"""

if __name__ == '__main__':
    logFile = 'translator-%s.log' % datetime.now().strftime("%Y-%m-%d_%H_%M")
    logLevel = logging.ERROR

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=logFile,
                        filemode='w')

    logger = logging.getLogger("MAIN")
    logger.setLevel(logLevel)

    parser = ArgumentParser(prog="translator.py", )

    subparsers = parser.add_subparsers(  # title='subcommands',
                                         # description='valid subcommands',
                                         # help='translator usage modes'
    )

    parser_prep = subparsers.add_parser('prepare', help='prepare translation data')
    parser_prep.add_argument('archive', type=str, help='ankiweb archive path')
    parser_prep.add_argument('file', type=str, help='path to file inside of archive')
    parser_prep.add_argument('dir', type=str, help='path to save preprocessed data')
    parser_prep.add_argument('-f', '--flip-translation-order', dest='flip', action='store_const', const=True, default=False, help='Flip translation order')
    parser_prep.add_argument('-l', '--limit-lines', type=int, help='Limit number of text lines to train on', default=50_000, required=False)
    parser_prep.add_argument('-s', '--vocab-size', type=int, help='Vectorization vocabulary size', default=50_000, required=False)
    parser_prep.add_argument('-q', '--sequence-length', type=int, help='Vectorization max sequence size', default=20, required=False)
    parser_prep.set_defaults(func=handle_prep)

    parser_create = subparsers.add_parser('create', help='create translation model')
    parser_create.add_argument('lang', type=str, help='path to language data')
    parser_create.add_argument('model', type=str, help='path to translation model')
    parser_create.add_argument('-b', '--batch-size', type=int, help='Training batch size', default=256, required=False)
    parser_create.add_argument('-e', '--embed-dim', type=int, help='embedding dim', default=256, required=False)
    parser_create.add_argument('-l', '--latent-dim', type=int, help='LSTM networks latent dim', default=1024, required=False)
    parser_create.add_argument('-n', '--num-heads', type=int, help='number of tracked heads for causal attention tracking', default=6, required=False)
    parser_create.add_argument('-d', '--dropout', type=float, help='dropout rate', default=0.125, required=False)
    parser_create.add_argument('-t', '--epochs', type=int, help='epochs to pre-train', default=4, required=False)
    parser_create.set_defaults(func=handle_create)

    parser_train = subparsers.add_parser('train', help='train model')
    parser_train.add_argument('lang', type=str, help='path to language data')
    parser_train.add_argument('model', type=str, help='path to translation model')
    parser_train.add_argument('epochs', type=int, help='epochs to train')
    parser_train.set_defaults(func=handle_train)

    parser_trans = subparsers.add_parser('translate', help='register user')
    parser_trans.add_argument('lang', type=str, help='path to language data')
    parser_trans.add_argument('model', type=str, help='path to translation model')
    parser_trans.add_argument('-p', '--phrase', type=str, help='phrase to translate', required=False)
    parser_trans.add_argument('-f', '--file', type=str, help='file to read phrases from', required=False)
    parser_trans.add_argument('-s', '--sample', type=int, help='number of elements from file to sample', required=False)
    parser_trans.set_defaults(func=handle_translate)

    if len(sys.argv) < 2:
        parser.print_help()
        parser_prep.print_usage()
        parser_create.print_usage()
        parser_train.print_usage()
        parser_trans.print_usage()
    else:
        args = parser.parse_args()
        args.func(args)

# Simple translator
Based on Keras-nlp

For theory behind operation look at the articles here: https://machinelearningmastery.com/category/attention/

# usage examples

## Initialize text data
`python3 transator.py prepare in/spa-eng.zip spa.txt lang-spa`

## Build translator model
`python3 transator.py create lang-spa trans-spa`

## Translate one phrase
`python3 transator.py translate lang-spa trans-spa -p "my father went fishing"`

## Translate test dataset
`python3 transator.py translate lang-spa trans-spa -f lang-spa/test_texts.txt`

# usage info
```
usage: translator.py [-h] {prepare,create,train,translate} ...

positional arguments:
  {prepare,create,train,translate}
    prepare             prepare translation data
    create              create translation model
    train               train model
    translate           register user

options:
  -h, --help            show this help message and exit

```
## `prepare` mode
```
usage: translator.py prepare [-h] [-f] [-l LIMIT_LINES] [-s VOCAB_SIZE] [-q SEQUENCE_LENGTH] archive file dir

positional arguments:
  archive               ankiweb archive path
  file                  path to file inside of archive
  dir                   path to save preprocessed data

options:
  -h, --help            show this help message and exit
  -f, --flip-translation-order
                        Flip translation order
  -l LIMIT_LINES, --limit-lines LIMIT_LINES
                        Limit number of text lines to train on
  -s VOCAB_SIZE, --vocab-size VOCAB_SIZE
                        Vectorization vocabulary size
  -q SEQUENCE_LENGTH, --sequence-length SEQUENCE_LENGTH
                        Vectorization max sequence size

```
## `create` mode
```
usage: translator.py create [-h] [-b BATCH_SIZE] [-e EMBED_DIM] [-l LATENT_DIM] [-n NUM_HEADS] [-d DROPOUT] [-t EPOCHS] lang model

positional arguments:
  lang                  path to language data
  model                 path to translation model

options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Training batch size
  -e EMBED_DIM, --embed-dim EMBED_DIM
                        embedding dim
  -l LATENT_DIM, --latent-dim LATENT_DIM
                        LSTM networks latent dim
  -n NUM_HEADS, --num-heads NUM_HEADS
                        number of tracked heads for causal attention tracking
  -d DROPOUT, --dropout DROPOUT
                        dropout rate
  -t EPOCHS, --epochs EPOCHS
                        epochs to pre-train

```
## `train` mode
```
usage: translator.py train [-h] lang model epochs

positional arguments:
  lang        path to language data
  model       path to translation model
  epochs      epochs to train

options:
  -h, --help  show this help message and exit

```
## `translate` mode
```
usage: translator.py translate [-h] [-p PHRASE] [-f FILE] lang model

positional arguments:
  lang                  path to language data
  model                 path to translation model

options:
  -h, --help            show this help message and exit
  -p PHRASE, --phrase PHRASE
                        phrase to translate
  -f FILE, --file FILE  file to read phrases from

```
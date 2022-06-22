#!/bin/bash
# init text data for model
python translator.py prepare in/spa-eng.zip spa.txt lang/spa-eng -l 30000 -s 30000
# create translator model
python translator.py create lang/spa-eng trans/spa-eng
# train translator model
python translator.py train lang/spa-eng trans/spa-eng 16

#!/bin/bash

python DoubleJetSLDA.py -L 8 -Ne 50

python DoubleJetSLDA.py -L 7 -Ne 118

python DoubleJetSLDA.py -L 6 -Ne 170

python DoubleJetMLDA.py -ls 7 8 -Ne 37 27

python DoubleJetMLDA.py -ls 6 7 8 -Ne 12 22 24
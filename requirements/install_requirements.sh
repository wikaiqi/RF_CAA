#!/bin/bash

#setup a conda envirnoment
conda create -n RFsleep   #  set the envirnoment name as RFsleep
source activate  RFsleep
conda install -c anaconda setuptools
pip install -r requirements.txt



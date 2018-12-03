# SA
This is the code about the arxiv paper "parameter-free spatial attention network for Person Re-Identification"

Our code is mainly based on PCB

# Preparation
Prerequisite: Python 2.7 and Pytorch 0.4.0

# Training
if you are going to train on the dataset of market-1501, run:

python2 main.py -d market -b 48 -j 4 --epochs 100 --log logs/market/ --combine-trainval --step-size 40 --data-dir Market-1501 

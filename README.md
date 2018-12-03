# SA
This is the code about the arxiv paper "parameter-free spatial attention network for Person Re-Identification"

# Training
if you are going to train on the dataset of CUHK03_labeled, run:

CUDA_VISIBLE_DEVICES=0 python2 main.py -d market -a resnet50 -b 48 -j 4 --epochs 100 --log logs/cuhk03_labeled/sgddrop0.7-x/ --combine-trainval --feature 256 --height 384 --width 128 --step-size 40 --data-dir /data1/whr/dataset/Market-1501 

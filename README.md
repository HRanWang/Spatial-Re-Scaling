# Parameter-Free Spatial Attention Network for Person Re-Identification
This is the implementation of the arxiv [paper](https://arxiv.org/abs/1811.12150) "Parameter-Free Spatial Attention Network for Person Re-Identification".

We propose a modification to the global average pooling called **spatial attention** which shows a consistent improvement in the generic classfication tasks. Currently the experiments are only conducted on the Person-ReID tasks (which is formulated into a fine-grained classification problem). Our code is mainly based on [PCB](https://github.com/syfafterzy/PCB_RPP_for_reID).

## Network
![](https://github.com/schizop/SA/blob/master/network/network.png) 

# Preparation
Prerequisite: Python 2.7 and Pytorch 0.4.0(we run the code under version 0.4.0, maybe versions <= 0.4.0 also work.)
## Dataset
[Market-1501](https://pan.baidu.com/s/1qlCJEdEY7UueGL-VdhH6xw) (password: 1ri5)
  
# Training
if you are going to train on the dataset of market-1501, run:

```
python2 main.py -d market -b 48 -j 4 --epochs 100 --log logs/market/ --combine-trainval --step-size 40 --data-dir Market-1501
```
also, you can just download a trained [weight file](https://pan.baidu.com/s/1mQkjrJMa1KQmaHax1kAjsw) from BaiduYun (password: wwjv)

# Results
![](https://github.com/schizop/SA/blob/master/results/result.jpg) 

# Citiaion

Please cite the paper if it helps your research:  
```
@article{wang2018parameter,
  title={Parameter-Free Spatial Attention Network for Person Re-Identification},
  author={Wang, Haoran and Fan, Yue and Wang, Zexin and Jiao, Licheng and Schiele, Bernt},
  journal={arXiv preprint arXiv:1811.12150},
  year={2018}
}
```

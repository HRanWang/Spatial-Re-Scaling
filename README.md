# Parameter-Free Spatial Attention Network for Person Re-Identification
This is the implementation of the arxiv [paper](https://arxiv.org/abs/1811.12150) "Parameter-Free Spatial Attention Network for Person Re-Identification".

We propose a modification to the global average pooling called **spatial attention** which shows a consistent improvement in the generic classfication tasks. Currently the experiments are only conducted on the Person-ReID tasks (which is formulated into a fine-grained classification problem). Our code is mainly based on [PCB](https://github.com/syfafterzy/PCB_RPP_for_reID).

## Network
![](https://github.com/schizop/SA/blob/master/network/network.png) 

The proposed architecture formulates the task as a **classification**. It consists of four components. The yellow region represents the
backbone feature extractor. The red region represents the deeply supervised branches (DS). The blue region represents six part classifiers
(P). The two green region represents two sets of spatial attention layers (SA), SA1 is not used for the main results. It only appears in
the ablation study. Then the total loss is the summation over all deep supervision losses, six part losses and the loss from the backbone. Note that the spatial attention is only added before GAP.

# Preparation
Prerequisite: Python 2.7 and Pytorch 0.4.0(we run the code under version 0.4.0, maybe versions <= 0.4.0 also work.)
## Dataset
[Market-1501](https://pan.baidu.com/s/1qlCJEdEY7UueGL-VdhH6xw) (password: 1ri5)
  
# Training&Testing
if you are going to train on the dataset of market-1501, run training:

```
python2 main.py -d market -b 48 -j 4 --epochs 100 --log logs/market/ --combine-trainval --step-size 40 --data-dir Market-1501
```
also, you can just download a trained [weight file](https://pan.baidu.com/s/1mQkjrJMa1KQmaHax1kAjsw) from BaiduYun (password: wwjv), and put it into model folder, which should be like 'model/checkpoint.pth.tar', then run testing:
```
python2 main.py -d market -b 48 -j 4 --log logs/market/ --combine-trainval --step-size 40 --data-dir Market-1501 --resume ./model/checkpoint.pth.tar --evaluate
```

# Results
![](https://github.com/schizop/SA/blob/master/results/sota.png) 

We achieved the state-of-the-art on four benchmarks as is shown in Table 1 (11. Nov. 2018).

![](https://github.com/schizop/SA/blob/master/results/result.jpg) 

Here we show 6 examples to compare the class activation maps ([CAM](https://arxiv.org/abs/1512.04150)) of plain GAP and GAP with SA. From left to right are the original image, the CAM from plain GAP and the CAM from GAP with SA. 

We see that the highlighted area from plain GAP is always concentrated to some parts of the object, which may suffer from the absence of that feature due to some occlusion and view point changing. With the help of the spatial attention, the focus of the model is distributed all over the image, providing the classifier more details of the object, which increases the model robustness.

# Ablation Study
In order to demonstrate the effectiveness of the spatial attention layer. We are now working on more examples for the ablation study. Each example inside the folder *ablation* is independent of the rest of the snippets.

## Person Re-ID:
Besides the ones in the paper, we uploaded another example for the ablation of the SA for the backbone model on Market-1501. Random erasing is cut off for the simplicity. The training epoch is set as 60.
```
python2 main.py -d market -b 48 -j 4 --epochs 60 --log logs/market/ --feature 256 --height 384 --width 128 --combine-trainval --step-size 40 --data-dir Market-1501
```
## Classification:
### Cifar 100:

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

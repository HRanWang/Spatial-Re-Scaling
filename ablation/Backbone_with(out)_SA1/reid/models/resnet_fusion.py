from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from torch import nn



class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization + relu.
    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class SpatialAttn(nn.Module):
    """Spatial Attention (Sec. 3.1.I.1)"""
    def __init__(self):
        super(SpatialAttn, self).__init__()

    def forward(self, x):
        # global cross-channel averaging # 32,2048,24,8
        x = x.mean(1, keepdim=True)  # 32,1,24,8
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0),-1)     # 32,192
        z = x
        for b in range(x.size(0)):
            # print z[b]
            z[b] /= torch.sum(z[b])
            # print z[b]
        z = z.view(x.size(0),1,h,w)
        return z


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True,
                 num_features=0, dropout=0, num_classes=0):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

#==========================add dilation=============================#
        for mo in self.base.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1,1)
#================append conv for FCN==============================#
        self.num_features = num_features
        self.num_classes = 751 #num_classes
        self.dropout = dropout
        out_planes = self.base.fc.in_features
        self.local_conv = nn.Conv2d(out_planes, self.num_features, kernel_size=1,padding=0,bias=False)
        self.local_conv_layer1 = nn.Conv2d(256, self.num_features, kernel_size=1,padding=0,bias=False)
        self.local_conv_layer2 = nn.Conv2d(512, self.num_features, kernel_size=1,padding=0,bias=False)
        self.local_conv_layer3 = nn.Conv2d(1024, self.num_features, kernel_size=1,padding=0,bias=False)
        self.local_conv_layer4 = nn.Conv2d(2048, self.num_features, kernel_size=1,padding=0,bias=False)

        init.kaiming_normal(self.local_conv.weight, mode= 'fan_out')
#            init.constant(self.local_conv.bias,0)
        self.feat_bn2d = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
        init.constant(self.feat_bn2d.weight,1) #initialize BN, may not be used
        init.constant(self.feat_bn2d.bias,0) # iniitialize BN, may not be used

        self.soft4 = SpatialAttn()

        # self.offset = ConvOffset2D(256)

        ##---------------------------stripe1----------------------------------------------#
        self.instance0 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance0.weight, std=0.001)
        init.constant(self.instance0.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance1 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance1.weight, std=0.001)
        init.constant(self.instance1.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance2 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance2.weight, std=0.001)
        init.constant(self.instance2.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance3 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance3.weight, std=0.001)
        init.constant(self.instance3.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance4 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance4.weight, std=0.001)
        init.constant(self.instance4.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance5 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance5.weight, std=0.001)
        init.constant(self.instance5.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance6 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance6.weight, std=0.001)
        init.constant(self.instance6.bias, 0)
# ##---------------------------stripe1----------------------------------------------#
#             self.instance7 = nn.Linear(self.num_features, self.num_classes)
#             init.normal(self.instance7.weight, std=0.001)
#             init.constant(self.instance7.bias, 0)
# ##---------------------------stripe1----------------------------------------------#
# ##---------------------------stripe1----------------------------------------------#
#             self.instance8 = nn.Linear(self.num_features, self.num_classes)
#             init.normal(self.instance8.weight, std=0.001)
#             init.constant(self.instance8.bias, 0)
# ##---------------------------stripe1----------------------------------------------#
# ##---------------------------stripe1----------------------------------------------#
#             self.instance9 = nn.Linear(self.num_features, self.num_classes)
#             init.normal(self.instance9.weight, std=0.001)
#             init.constant(self.instance9.bias, 0)
# ##---------------------------stripe1----------------------------------------------#
# ##---------------------------stripe1----------------------------------------------#
#             self.instance10 = nn.Linear(self.num_features, self.num_classes)
#             init.normal(self.instance10.weight, std=0.001)
#             init.constant(self.instance10.bias, 0)
# ##---------------------------stripe1----------------------------------------------#
# ##---------------------------stripe1----------------------------------------------#
#             self.instance11 = nn.Linear(self.num_features, self.num_classes)
#             init.normal(self.instance11.weight, std=0.001)
#             init.constant(self.instance11.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance_layer1 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance_layer1.weight, std=0.001)
        init.constant(self.instance_layer1.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance_layer2 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance_layer2.weight, std=0.001)
        init.constant(self.instance_layer2.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
        self.instance_layer3 = nn.Linear(self.num_features, self.num_classes)
        init.normal(self.instance_layer3.weight, std=0.001)
        init.constant(self.instance_layer3.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#


        self.fusion_conv = nn.Conv1d(4, 1, kernel_size=1, bias=False)

        self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
            if name == 'layer1':
                x_layer1 = x
                continue
            if name == 'layer2':
                x_layer2 = x
            if name == 'layer3':
                x_layer3 = x
                # f3 = x


        sx = x.size(2) / 6
        kx = x.size(2) - sx * 5

        x = F.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))  # H4 W8
        # x_attn4 = self.soft4(x)          ################  Add SA1 here when do ablation study !!!!
        # x = x*x_attn4
        # print sx,kx,x.size()
        # ========================================================================#

        out0 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        x = self.drop(x)
        x = self.local_conv(x)
        out1 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        x = self.feat_bn2d(x)
        x = F.relu(x)  # relu for local_conv feature
        # x = self.drop(x)

        x6 = F.avg_pool2d(x, kernel_size=(6, 1), stride=(1, 1))

        x6 = x6.contiguous().view(x6.size(0), -1)

        c6 = self.instance6(x6)

        return out0, c6

#==========================================================#
# =======================DCN===============================#
#         if self.DCN:
#             y = x.unsqueeze(1)
#             y = F.avg_pool3d(x, (16, 1, 1)).squeeze(1)
#             sx = x.size(2) / 6
#             kx = x.size(2) - sx * 5
#             x = F.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))  # H4 W8
#             # print sx,kx,x.size()
#             # ========================================================================#
#
#             out0 = x.view(x.size(0), -1)
#             out0 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
#             x = self.drop(x)
#             x = self.local_conv(x)
#             x = self.offset(x)
#             out1 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
#             x = self.feat_bn2d(x)
#             x = F.relu(x)  # relu for local_conv feature
#
#             x = x.chunk(6, 2)
#             x0 = x[0].contiguous().view(x[0].size(0), -1)
#             x1 = x[1].contiguous().view(x[1].size(0), -1)
#             x2 = x[2].contiguous().view(x[2].size(0), -1)
#             x3 = x[3].contiguous().view(x[3].size(0), -1)
#             x4 = x[4].contiguous().view(x[4].size(0), -1)
#             x5 = x[5].contiguous().view(x[5].size(0), -1)
#             c0 = self.instance0(x0)
#             c1 = self.instance1(x1)
#             c2 = self.instance2(x2)
#             c3 = self.instance3(x3)
#             c4 = self.instance4(x4)
#             c5 = self.instance5(x5)
#             return out0, (c0, c1, c2, c3, c4, c5)

# ==========================================================#
	

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

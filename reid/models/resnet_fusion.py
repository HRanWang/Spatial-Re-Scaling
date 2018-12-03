from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self,block,layers, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, FCN=False, radius=1., thresh=0.5):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.FCN = FCN

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        # ==========================add dilation=============================#
        if self.FCN:
            self.num_features = num_features
            self.num_classes = 751  # num_classes
            self.dropout = dropout
            # out_planes = self.base.fc.in_features
            self.local_conv1 = nn.Conv2d(256, self.num_features, kernel_size=1, padding=0, bias=False)
            self.local_conv2 = nn.Conv2d(512, self.num_features, kernel_size=1, padding=0, bias=False)
            self.local_conv3 = nn.Conv2d(1024, self.num_features, kernel_size=1, padding=0, bias=False)
            self.local_conv4 = nn.Conv2d(2048, self.num_features, kernel_size=1, padding=0, bias=False)

            init.kaiming_normal(self.local_conv1.weight, mode='fan_out')
            init.kaiming_normal(self.local_conv2.weight, mode='fan_out')
            init.kaiming_normal(self.local_conv3.weight, mode='fan_out')
            init.kaiming_normal(self.local_conv4.weight, mode='fan_out')

            #            init.constant(self.local_conv.bias,0)
            self.feat_bn2d = nn.BatchNorm2d(self.num_features)  # may not be used, not working on caffe
            init.constant(self.feat_bn2d.weight, 1)  # initialize BN, may not be used
            init.constant(self.feat_bn2d.bias, 0)  # iniitialize BN, may not be used

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
            init.normal(self.instance5.weight, std=0.001)
            init.constant(self.instance5.bias, 0)

            self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # for name, module in self.base._modules.items():
        #     if name == 'avgpool':
        #         break
        #     x = module(x)
        #
        # if self.cut_at_pooling:
        #     return x
        # =======================FCN===============================#
        if self.FCN:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            stage1 = self.layer1(x)
            stage2 = self.layer2(stage1)
            stage3 = self.layer3(stage2)
            stage4 = self.layer4(stage3)
            print stage1.size()
            print stage2.size()
            print stage3.size()
            print stage4.size()

            # z = stage4[:, :, 5:17, :]
            # z = self.drop(z)
            # z = self.local_conv(z)
            # z = self.feat_bn2d(z)
            # z = F.relu(z)  # relu for local_conv feature
            # x6 = F.avg_pool2d(z, kernel_size=(12, 8), stride=(1, 1))
            # x6 = x6.contiguous().view(x6.size(0), -1)
            # # print x6.size()
            # c6 = self.instance6(x6)

            z = stage4
            sx = z.size(2) / 6
            kx = z.size(2) - sx * 5
            z = F.avg_pool2d(z, kernel_size=(kx, z.size(3)), stride=(sx, z.size(3)))  # H4 W8
            out0 = z/z.norm(2,1).unsqueeze(1).expand_as(z)

            stages = [stage1,stage2,stage3,stage4]
            losses = []
            num = 0
            for stage in stages:
                x = stage
                sx = x.size(2) / 6
                kx = x.size(2) - sx * 5
                x = F.avg_pool2d(x, kernel_size=(kx, x.size(3)),
                                 stride=(sx, x.size(3)))  # H4 W8   #torch.Size([64, 2048, 6, 1])
                # print sx, kx, x.size()
                # ========================================================================#
                x = self.drop(x)
                # x = self.local_conv(x)
                # x = self.local_convs[num](x)
                num+=1
                if num == 1:
                    x = self.local_conv1(x)
                elif num == 2:
                    x = self.local_conv2(x)
                elif num == 3:
                    x = self.local_conv3(x)
                elif num == 4:
                    x = self.local_conv4(x)
                x = self.feat_bn2d(x)
                x = F.relu(x)  # relu for local_conv feature
                x = x.chunk(6, 2)
                # print x[0].size()

                x0 = x[0].contiguous().view(x[0].size(0), -1)
                x1 = x[1].contiguous().view(x[1].size(0), -1)
                x2 = x[2].contiguous().view(x[2].size(0), -1)
                x3 = x[3].contiguous().view(x[3].size(0), -1)
                x4 = x[4].contiguous().view(x[4].size(0), -1)
                x5 = x[5].contiguous().view(x[5].size(0), -1)

                # print x0.size()
                c0 = self.instance0(x0)
                # print c0.size()
                c1 = self.instance1(x1)
                c2 = self.instance2(x2)
                c3 = self.instance3(x3)
                c4 = self.instance4(x4)
                c5 = self.instance5(x5)


                # print c6.size()
                losses.append(c0)
                losses.append(c1)
                losses.append(c2)
                losses.append(c3)
                losses.append(c4)
                losses.append(c5)
            # losses.append(out0)
            return out0,losses

        # ==========================================================#
        # =======================DCN===============================#
        #         if self.FCN:
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

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        out1 = x.view(x.size(0), -1)
        center = out1.mean(0).unsqueeze(0).expand_as(out1)
        out2 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)

        if self.has_embedding:
            x = self.feat(x)
            out3 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
            x = self.feat_bn(x)

        if self.norm:
            x = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        elif self.has_embedding:  # adding relu after fc, not used in softmax but in tripletloss
            x = F.relu(x)
            out4 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return out2, x, out2, out2

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

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}
def resnet50(**kwargs):
    # return ResNet(Bottleneck, [3, 4, 6, 3],50, **kwargs)
    model = ResNet(Bottleneck, [3, 4, 6, 3],50, **kwargs)
    print('load resnet50 ok')
    model_dict = model.state_dict()
    params = model_zoo.load_url(model_urls['resnet50'])
    params = {k: v for k, v in params.items() if k in model_dict}
    model_dict.update(params)
    model.load_state_dict(model_dict)
    return model
def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)

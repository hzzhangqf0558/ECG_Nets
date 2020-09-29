import torch
import torch.nn as nn
from torchsummary import summary
from models.BasicModule import BasicModule
import math
import torch.utils.model_zoo as model_zoo


#declare
__all__ = ['SENet', 'se_resnet_b_18', 'se_resnet_b_34', 'se_resnet_b_50', 'se_resnet_b_101',
           'se_resnet_b_152']

def conv1x3(in_planes, out_planes, stride=1):
    """1x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        #head conv1 7x7, 64 stride=2
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.5)

         #### update -----
        # 64x2500, 128x625, 256x313, 512x157
        self.globalAvgPool = nn.AdaptiveAvgPool1d(1)
        # self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        # self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        # SE layers
        self.fc1 = nn.Conv1d(planes, planes // 16, kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
        self.fc2 = nn.Conv1d(planes // 16, planes, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)   #dropout
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        original_out = out
        out = self.globalAvgPool(out)
        #out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        #out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #1x1 64->3x3 64->1x1 4*64
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        #four groups 64x56x56, 128x28x28,512x7x7  modify --

        self.globalAvgPool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Conv1d(planes * 4, round(planes / 4), kernel_size=1)  # Use nn.Conv1d instead of nn.Linear
        self.fc2 = nn.Conv1d(round(planes / 4), planes * 4, kernel_size=1)

        # self.fc1 = nn.Linear(in_features=planes * 4, out_features=round(planes / 4))
        # self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes * 4)
        self.sigmoid = nn.Sigmoid()
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

        original_out = out
        out = self.globalAvgPool(out)
        # out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        # out = out.view(out.size(0),out.size(1),1,1)
        out = out * original_out

        out += residual
        out = self.relu(out)

        return out


class SENet(BasicModule):

    def __init__(self, block, layers, num_classes=55):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.conv1 = nn.Conv1d(8, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Baseline
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.sigmoid = nn.Sigmoid()  # multi-task
        self.init()

        #init weights biases
        # for layer in self.modules():
        #     if isinstance(layer, nn.Conv2d):
        #         nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(layer, nn.BatchNorm2d):
        #         nn.init.constant_(layer.weight, 1)
        #         nn.init.constant_(layer.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.sigmoid(x)  #multi-task

        return x


def se_resnet_b_18(num_classes=55):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(BasicBlock, [2, 2, 2, 2], num_classes=55)
    return model


def se_resnet_b_34(num_classes=55):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(BasicBlock, [3, 4, 6, 3], num_classes)
    return model


def se_resnet_b_50(num_classes=55):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 4, 6, 3], num_classes)
    return model


def se_resnet_b_101(num_classes=55):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


def se_resnet_b_152(num_classes=55):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 8, 36, 3], num_classes)
    return model

def test_se_resNet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = se_resnet_b_50(34)
    model = net.to(device)
    print(summary(net, input_size=(8, 5000)))


if __name__ == '__main__':
    test_se_resNet()
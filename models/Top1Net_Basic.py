import torch
import torch.nn as nn
from torchsummary import summary
from models.BasicModule import BasicModule

import torch.nn.functional as F

   #block1  = bn--relu--conv
class Block1(BasicModule):
    def __init__(self, in_planes, planes, kernel_size=[1, 15], stride=[1, 2], padding=[0, 7]):
        super(Block1, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.shortcut = nn.Sequential()

        if stride != [1, 1] or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//4, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//4, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)

        # Squeeze
        w = F.avg_pool2d(out, kernel_size=[out.size(2), out.size(3)])
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)

        return out


#block2 = bn--relu--conv x3
class Block2(BasicModule):
    def __init__(self, in_planes, planes, kernel_size=[1, 3], stride=[1, 1], padding=[0, 1]):
        super(Block2, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.bn3 = nn.BatchNorm2d(in_planes)
        self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.shortcut = nn.Sequential()

        if stride != [1, 1] or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//4, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//4, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)

        out = F.relu(self.bn2(out))
        out = self.conv2(out)

        out = F.relu(self.bn3(out))
        out = self.conv3(out)

        # Squeeze
        w = F.avg_pool2d(out, kernel_size=[out.size(2), out.size(3)])
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)

        return out



#block3 = bn1--relu1--conv1-SE x3  bn1--relu1--conv1-SE x3
class Block3(BasicModule):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1):
        super(Block3, self).__init__()

        #the first
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.bn3 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding=padding, stride=2, bias=False)




        self.shortcut1 = nn.Sequential(
            nn.BatchNorm1d(in_planes),
            nn.ReLU(),
            nn.Conv1d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=2, bias=False)
        )

        # the first SE layers
        self.fc1 = nn.Conv1d(planes, planes//4, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv1d(planes//4, planes, kernel_size=1)



        # the second

        # the first
        self.bn4 = nn.BatchNorm1d(planes)
        self.conv4 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.bn5 = nn.BatchNorm1d(planes)
        self.conv5 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.bn6 = nn.BatchNorm1d(planes)
        self.conv6 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)

        self.shortcut2 = nn.Sequential(
            nn.BatchNorm1d(in_planes),
            nn.ReLU(),
            nn.Conv1d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        )

        # the second SE
        self.fc3 = nn.Conv1d(planes, planes // 4, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc4 = nn.Conv1d(planes // 4, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)

        out = F.relu(self.bn2(out))
        out = self.conv2(out)

        out = F.relu(self.bn3(out))
        out = self.conv3(out)

        # Squeeze
        w = F.avg_pool1d(out, kernel_size=out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!
        out += self.shortcut1(x)

        origin = out

        # the second
        out = F.relu(self.bn4(out))
        out = self.conv4(out)

        out = F.relu(self.bn5(out))
        out = self.conv5(out)

        out = F.relu(self.bn6(out))
        out = self.conv6(out)

        # Squeeze
        w = F.avg_pool1d(out, kernel_size=out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!
        out += self.shortcut2(origin)
        return out



class Top1Net(BasicModule):
    def __init__(self, num_classes=55):
        super(Top1Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=[1, 50], stride=[1, 2], padding=[0, 0], bias=False) # output:2476
        self.bn1 = nn.BatchNorm2d(32)

        self.inplanes = 32
        self.layers_block1s = self._make_layer(Block1, self.inplanes, 32, 3, kernel_size=[1, 15], stride=[1, 2], padding=[0, 7])

        #kernel=3,5,7 configurations
        self.sizes = [3,5,7]
        self.strides = [1,1,1]
        self.pads = [1,2,3]

        # self.sizes = [3, 5, 7, 9]
        # self.strides = [1, 1, 1, 1]
        # self.pads = [1, 2, 3, 4]

        self.layer_block2s_list = []
        self.layer2_block3s_list = []
        for i in range(len(self.sizes)):

            layers_block2s = self._make_layer(Block2, self.inplanes, self.inplanes, 4, kernel_size=[1, self.sizes[i]], stride=[1, self.strides[i]],
                                                   padding=[0, self.pads[i]])
            self.layer_block2s_list.append(layers_block2s)

            layers_block3s = self._make_layer(Block3, self.inplanes * 8, self.inplanes*8, 4, kernel_size=self.sizes[i], stride=self.strides[i],
                                                   padding=self.pads[i])
            self.layer2_block3s_list.append(layers_block3s)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(256 * len(self.sizes), num_classes)  #fully connected
        self.sigmoid = nn.Sigmoid()  # multi-task

    def forward(self, x0):
        x0 = x0.unsqueeze(1)
        x0 = F.relu(self.bn1(self.conv1(x0)))
        x0 = self.layers_block1s(x0)

        xs = []

        for i in range(len(self.sizes)):
            x = self.layer_block2s_list[i](x0)
            x = torch.flatten(x, start_dim=1, end_dim=2)
            x = self.layer2_block3s_list[i](x)
            x = self.avgpool(x)
            xs.append(x)

        out = torch.cat(xs,dim=2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out



    def _make_layer(self, block, inplanes, planes, blocks, kernel_size, stride, padding):
        layers = []
        for i in range(blocks):
            layers.append(block(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding))
        return nn.Sequential(*layers)

def Top1Net_b_25( num_classes=55):
    """
        SE top1net
    """
    model = Top1Net(num_classes=num_classes)
    return model


def test_se_resNet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # net = Block1(1,32,kernel_size=[1, 15], stride=[1, 2], padding=[0, 7])
    # model = net.to(device)
    # # print(summary(net, input_size=( 1, 8, 1238)))
    # # y =net(torch.randn(32, 1, 8, 5000))
    # # print(y.size())
    # sizes = [3,4]
    # print([1,sizes[1]])
    # for i in range(10):
    #     print(i)

    net = Top1Net_b_25(num_classes=55)
    model = net.to(device)
    y =net(torch.randn(32, 8, 5000))
    print(y.size())
    print(summary(net, input_size=(8, 5000)))

    # net = Block2(32,32,kernel_size=[1, 3], padding=[0,1],stride=[1,1])
    # model = net.to(device)
    # print(summary(net,input_size=(32, 8, 310)))

    # net = Block3(256,256,kernel_size=3, padding=1,stride=1)
    # model = net.to(device)
    # print(summary(net,input_size=(256, 20)))


if __name__ == '__main__':
    test_se_resNet()
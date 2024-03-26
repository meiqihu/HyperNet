import torch.nn as nn
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



# 常见的3x3卷积
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# return  self.sigmoid(out)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, inplane):
        super(CBAM, self).__init__()
        # self.ca = LocalChannelAttention(inplane)  # N, 32, H,W
        self.ca = ChannelAttention(inplane)
        self.sa = SpatialAttention()
        self.sa_weight = 0

    def forward(self, x):
        x = self.ca(x) * x
        self.sa_weight = self.sa(x)
        x = self.sa_weight * x
        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=True):  # inplanes代表输入通道数，planes代表输出通道数。
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        if use_cbam:
            self.cbam = CBAM(planes)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)
        out += residual
        out = self.relu(out)

        return out


class HyperNet(nn.Module):
    def __init__(self, block, layernum, gamma):
        # layernum: [127, 64, 32]
        super(HyperNet, self).__init__()
        if layernum is None:
            layernum = [127, 64, 128, 64]
        self.conv1 = self._make_layer(block, layernum[0], layernum[1])
        self.conv2 = self._make_layer(block, layernum[1], layernum[1])
        self.conv3 = self._make_layer(block, layernum[1], layernum[1])
        self.fc1 = FC(layernum[0], layernum[1])
        self.fc2 = FC(layernum[1], layernum[1])
        self.fc3 = FC(layernum[1], layernum[1])

        self.fuse1 = nn.Sequential(nn.Conv2d(layernum[1], layernum[1], kernel_size=1, padding=0, bias=False),
                                  nn.BatchNorm2d(layernum[1]))
        self.fuse2 = nn.Sequential(nn.Conv2d(layernum[1], layernum[1], kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(layernum[1]))

        self.proj = projector_C(feature_num=layernum[2])
        self.pred = predictor_C(feature_num=[layernum[2], layernum[3]])

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.F_cos = Focal_cos(gamma)

    def _make_layer(self, block, inplanes, planes):
        downsample = None
        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes))
        layers = []
        layers.append(block(inplanes, planes, stride=1, downsample=downsample, use_cbam=True))
        return nn.Sequential(*layers)

    def spatial(self, x):
        a1 = self.conv1(x)
        a1 = self.conv2(a1)
        a1 = self.avgpool(self.conv3(a1))
        return a1
    def sprectral(self, x):
        b1 = self.fc3(self.fc2(self.fc1(x)))
        return b1

    def test(self, X1, X2):
        f1 = torch.cat([self.fuse1(self.spatial(X1)), self.fuse2(self.sprectral(X1))], dim=1)
        f2 = torch.cat([self.fuse1(self.spatial(X2)), self.fuse2(self.sprectral(X2))], dim=1)
        return f1.detach().cpu(), f2.detach().cpu()

    def forward(self, X1, X2, idx):
        if self.training:
            #  fuse(avgpool(spatial) + spectral)
            z1 = torch.cat([self.fuse1(self.spatial(X1)), self.fuse2(self.sprectral(X1))], dim=1)
            z1 = self.proj(z1)
            z2 = torch.cat([self.fuse1(self.spatial(X2)), self.fuse2(self.sprectral(X2))], dim = 1)
            z2 = self.proj(z2)
            p1 = self.pred(z1)  # [N, 24, H, W]
            p2 = self.pred(z2)  # [N, 24, H, W]

            z1 = z1.permute([0, 2, 3, 1]).view([-1, z1.shape[1]])  # [H*W, 24]
            z2 = z2.permute([0, 2, 3, 1]).view([-1, z2.shape[1]])  # [H*W, 24]
            p1 = p1.permute([0, 2, 3, 1]).view([-1, p1.shape[1]])  # [H*W, 24]
            p2 = p2.permute([0, 2, 3, 1]).view([-1, p2.shape[1]])  # [H*W, 24]
            L = self.F_cos(p1, p2, z1, z2, idx)
            return L
        else:
            return self.test(X1, X2)


class FC(nn.Module):
    def __init__(self, inplanes, planes):
        super(FC, self).__init__()
        self.fc = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(planes))
        self.ca = ChannelAttention_(planes)
        self.inplanes = inplanes
        self.planes = planes
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.ca(self.fc(x))
        if self.inplanes == self.planes:
            output = self.relu(x + output)
        return output
# return  x*self.sigmoid(out)
class ChannelAttention_(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x*self.sigmoid(out)

class projector_C(nn.Module):
    def __init__(self, feature_num):
        super(projector_C, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(feature_num, feature_num, kernel_size=(1, 1), bias=False),
                                  nn.BatchNorm2d(feature_num),
                                  nn.ReLU(inplace=True),  # first layer
                                  nn.Conv2d(feature_num, feature_num, kernel_size=(1, 1), bias=False),
                                  nn.BatchNorm2d(feature_num),
                                  nn.ReLU(inplace=True),  # second layer
                                  nn.Conv2d(feature_num, feature_num, kernel_size=(1, 1), bias=True),
                                  nn.BatchNorm2d(feature_num, affine=False))  # output layer

    def forward(self, x):
        return self.proj(x)

class predictor_C(nn.Module):
    def __init__(self, feature_num):
        super(predictor_C, self).__init__()
        self.pred = nn.Sequential(nn.Conv2d(feature_num[0], feature_num[1], kernel_size=(1, 1), bias=False),
                                  nn.BatchNorm2d(feature_num[1]),
                                  nn.ReLU(inplace=True),  # hidden layer
                                  nn.Conv2d(feature_num[1], feature_num[0], kernel_size=(1, 1),
                                            bias=False))  # output layer
    def forward(self, x):
        return self.pred(x)


class Focal_cos(nn.Module):
    def __init__(self, gamma=1):
        super(Focal_cos, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.gamma = gamma

    def forward(self, p1, p2, z1, z2, idx):
        cos_ = self.cos(p1, z2.detach())  # [H*W, 1]
        loss1 = torch.mul(torch.pow((2 - cos_), self.gamma), cos_)
        cos_ = self.cos(p2, z1.detach())  # [H*W, 1]
        loss2 = torch.mul(torch.pow((2 - cos_), self.gamma), cos_)
        L = -(loss1.index_select(0, idx).mean() + loss2.index_select(0, idx).mean()) * 0.5
        return L

def zz():
    print('Hello!')

if __name__ == "__main__":

    # block, layernum, gamma
    layernum = [127, 64, 128, 64]
    m = HyperNet(BasicBlock, layernum, 2)
    x1 = torch.randn([1, 127, 450, 375])
    x2 = torch.randn([1, 127, 450, 375])
    idx = torch.tensor([1, 2, 3, 4])
    # output = m(x1,x2,idx,'train')
    # print(output.shape)

    f1, f2 = m(x1, x2, idx, 'valid')
    print(f1.shape)



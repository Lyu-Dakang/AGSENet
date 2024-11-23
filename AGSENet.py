import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from REU_1 import CIM
from REU_1 import CIM_L
from HLE import HLE
__all__ = ['AGSENet']


class REBNCONV(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src,tar):

    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

    return src


### RSU-7 ###
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)

        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool4 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.pool5 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)


        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self,x):

        hx = x.float()
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)

        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = _upsample_like(hx6d,hx5)

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):

        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


##### AGSENet ####
class AGSENet(nn.Module):

    def __init__(self, num_classes, in_ch=3):
        super(AGSENet, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 32)
        self.wra1 = CIM_L(32)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(32, 16, 32)
        self.wra2 = CIM_L(32)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(32, 16, 32)
        self.wra3 = CIM_L(32)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(32, 16, 32)
        self.wra4 = CIM_L(32)
        self.pool45 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(32, 16, 64)
        self.wra5 = CIM(64)
        self.pool56 = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)
        self.wra6 = CIM(64)

        self.sse0 = HLE(32)
        self.sse1 = HLE(32)
        self.sse2 = HLE(32)
        self.sse3 = HLE(32)
        self.sse4 = HLE(64)

        # decoder
        self.stage5d = RSU4F(128, 16, 32)
        self.stage4d = RSU4(64, 16, 32)
        self.stage3d = RSU5(64, 16, 32)
        self.stage2d = RSU6(64, 16, 32)
        self.stage1d = RSU7(64, 16, 32)

        self.side1 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.side2 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.side3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.side4 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.side5 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.side6 = nn.Conv2d(64, num_classes, 3, padding=1)
        self.outconv = nn.Conv2d(6 * num_classes, num_classes, 1)

    def forward(self,x):

        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx1 = self.wra1(hx1)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx2 = self.wra2(hx2)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx3 = self.wra3(hx3)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx4 = self.wra4(hx4)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx5 = self.wra5(hx5)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx61 = self.wra6(hx6)
        hx6up = _upsample_like(hx61, hx5)

        # -------------------- decoder --------------------
        hx51 = self.sse4(hx5, hx6up)
        hx5d = self.stage5d(hx51)
        hx5dup = _upsample_like(hx5d, hx4)

        hx41 = self.sse3(hx4, hx5dup)
        hx4d = self.stage4d(hx41)
        hx4dup = _upsample_like(hx4d, hx3)

        hx31 = self.sse2(hx3, hx4dup)
        hx3d = self.stage3d(hx31)
        hx3dup = _upsample_like(hx3d, hx2)

        hx21 = self.sse1(hx2, hx3dup)
        hx2d = self.stage2d(hx21)
        hx2dup = _upsample_like(hx2d, hx1)

        hx11 = self.sse0(hx1, hx2dup)
        hx1d = self.stage1d(hx11)

        # side output
        output1 = self.side1(hx1d)

        output2 = self.side2(hx2d)
        output2 = _upsample_like(output2, output1)

        output3 = self.side3(hx3d)
        output3 = _upsample_like(output3, output1)

        output4 = self.side4(hx4d)
        output4 = _upsample_like(output4, output1)

        output5 = self.side5(hx5d)
        output5 = _upsample_like(output5, output1)

        output6 = self.side6(hx6)
        output6 = _upsample_like(output6, output1)

        output0 = self.outconv(torch.cat((output1, output2, output3, output4, output5, output6), 1))

        return output0, output1, output2, output3, output4, output5, output6


import torch
import torch.nn as nn
import torchvision.models as models
from model.ResNet import B2_ResNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn import Parameter, Softmax
import numpy as np
from model.HolisticAttention import HA
import torch.nn.functional as F
from collections import OrderedDict



class DynDiscriminator(nn.Module):
    def __init__(self, ndf = 64):
        super(DynDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x



class DynDiscriminatorSOD(nn.Module):
    def __init__(self, ndf = 64):
        super(DynDiscriminatorSOD, self).__init__()
        self.conv1_1 = nn.Conv2d(7, ndf, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(ndf)
        self.bn1_2 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x, pred):
        x = torch.cat((x, pred), 1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x




class DynDiscriminatorCOD(nn.Module):
    def __init__(self, ndf = 64):
        super(DynDiscriminatorCOD, self).__init__()
        self.conv1_1 = nn.Conv2d(7, ndf, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(ndf)
        self.bn1_2 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x, pred):
        x = torch.cat((x, pred), 1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x




class FCDiscriminator(nn.Module):
    def __init__(self, ndf = 64):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x



class FCDiscriminatorsod(nn.Module):
    def __init__(self, ndf = 64):
        super(FCDiscriminatorsod, self).__init__()
        self.conv1 = nn.Conv2d(1, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x


class Generator(nn.Module):
    def __init__(self, channel=32):
        super(Generator, self).__init__()

        self.sal_encoder = Saliency_feat_encoder()
        self.cod_encoder = Cod_feat_encoder()
        self.shared_decoder = Share_feat_decoder(channel)
    def forward(self, x, is_sal):
        if is_sal:
            x1, x2, x3, x4 = self.sal_encoder(x)
            self.sal_init, self.sal_ref = self.shared_decoder(x1, x2, x3, x4)
            return  self.sal_init,self.sal_ref
        else:
            x1,x2,x3,x4 = self.cod_encoder(x)
            self.cod_init,self.cod_ref = self.shared_decoder(x1, x2, x3, x4)
            return self.cod_init,self.cod_ref



class CAM_Module(nn.Module):
    """ Channel attention module"""
    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class PAM_Module(nn.Module):
    """ Position attention module"""
    #paper: Dual Attention Network for Scene Segmentation
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class Triple_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Triple_Conv, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)



class Share_feat_decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32):
        super(Share_feat_decoder, self).__init__()
        self.resnet = B2_ResNet()
        self.relu = nn.ReLU(inplace=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.dropout = nn.Dropout(0.3)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2048)
        self.layer6 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 3)

        # self.conv1 = nn.Conv2d(256, channel, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(512, channel, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(1024, channel, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(2048, channel, kernel_size=1, padding=0)
        self.conv4_2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

        # self.conv1 = Triple_Conv(256, channel)
        # self.conv2 = Triple_Conv(512, channel)
        # self.conv3 = Triple_Conv(1024, channel)
        # self.conv4 = Triple_Conv(2048, channel)

        self.conv_feat = nn.Conv2d(32 * 5, channel, kernel_size=3, padding=1)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.pam_attention5 = PAM_Module(channel)
        self.pam_attention4 = PAM_Module(channel)
        self.pam_attention3 = PAM_Module(channel)
        self.pam_attention2 = PAM_Module(channel)

        # self.cam_attention4 = CAM_Module()
        # self.cam_attention3 = CAM_Module()
        # self.cam_attention2 = CAM_Module()

        self.pam_attention1 = PAM_Module(channel)
        self.racb_layer = RCAB(channel * 4)

        self.conv4 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.conv1 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 256)

        self.racb_43 = RCAB(channel * 2)
        self.racb_432 = RCAB(channel * 3)
        self.racb_4321 = RCAB(channel * 4)

        self.conv43 = Triple_Conv(2 * channel, channel)
        self.conv432 = Triple_Conv(3 * channel, channel)
        self.conv4321 = Triple_Conv(4 * channel, channel)

        self.HA = HA()
        self.conv4_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 2048)
        self.conv3_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 1024)
        self.conv2_2 = self._make_pred_layer(Classifier_Module, [3, 6, 12, 18], [3, 6, 12, 18], channel, 512)
        self.pam_attention4_2 = PAM_Module(channel)
        self.pam_attention3_2 = PAM_Module(channel)
        self.pam_attention2_2 = PAM_Module(channel)
        # self.cam_attention4_2 = CAM_Module()
        # self.cam_attention3_2 = CAM_Module()
        # self.cam_attention2_2 = CAM_Module()
        self.racb_43_2 = RCAB(channel * 2)
        self.racb_432_2 = RCAB(channel * 3)
        self.conv43_2 = Triple_Conv(2 * channel, channel)
        self.conv432_2 = Triple_Conv(3 * channel, channel)
        self.conv4321_2 = Triple_Conv(4 * channel, channel)
        self.layer7 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel * 4)

        if self.training:
            self.initialize_weights()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x1, x2, x3, x4):
        # edge( b, 1, 88, 88)
        conv1_feat = self.conv1(x1)
        conv2_feat = self.conv2(x2)
        conv2_feat1 = self.pam_attention2(conv2_feat)
        # conv2_feat2 = self.cam_attention2(conv2_feat)
        # conv2_feat = conv2_feat1 + conv2_feat2
        conv2_feat = conv2_feat1 
        conv3_feat = self.conv3(x3)
        conv3_feat1 = self.pam_attention3(conv3_feat)
        # conv3_feat2 = self.cam_attention3(conv3_feat)
        # conv3_feat = conv3_feat1 + conv3_feat2
        conv3_feat = conv3_feat1 
        conv4_feat = self.conv4(x4)
        conv4_feat1 = self.pam_attention4(conv4_feat)
        # conv4_feat2 = self.cam_attention4(conv4_feat)
        # conv4_feat = conv4_feat1 + conv4_feat2
        conv4_feat = conv4_feat1
        conv4_feat = self.upsample2(conv4_feat) #(b, 32, 22, 22)
        conv43 = torch.cat((conv4_feat, conv3_feat), 1) # (b, 65, 22, 22)
        conv43 = self.racb_43(conv43)
        conv43 = self.conv43(conv43)
        conv43 = self.upsample2(conv43) #(b, 32, 44, 44)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432(conv432)  #(b, 97, 44, 44)

        pred_init = self.layer6(conv432) #(b, 1, 44, 44)



        x2_2 = self.HA(pred_init.sigmoid(), x2) #(b, 512, 44, 44)
        x3_2 = self.resnet.layer3_2(x2_2)  # 1024 x 22 x 22
        x4_2 = self.resnet.layer4_2(x3_2)  # 2048 x 11 x 11

        conv2_feat = self.conv2_2(x2_2)
        conv2_feat1 = self.pam_attention2_2(conv2_feat)
        # conv2_feat2 = self.cam_attention2_2(conv2_feat)
        # conv2_feat = conv2_feat1+conv2_feat2
        conv2_feat = conv2_feat1
        conv3_feat = self.conv3_2(x3_2)
        conv3_feat1 = self.pam_attention3_2(conv3_feat)
        # conv3_feat2 = self.cam_attention3_2(conv3_feat)
        # conv3_feat = conv3_feat1+conv3_feat2
        conv3_feat = conv3_feat1
        conv4_feat = self.conv4_2(x4_2)
        conv4_feat1 = self.pam_attention4_2(conv4_feat)
        # conv4_feat2 = self.cam_attention4_2(conv4_feat)
        # conv4_feat = conv4_feat1+conv4_feat2
        conv4_feat = conv4_feat1
        conv4_feat = self.upsample2(conv4_feat)

        conv43 = torch.cat((conv4_feat, conv3_feat), 1) # (b, 64, 22, 22)
        conv43 = self.racb_43_2(conv43)
        conv43 = self.conv43_2(conv43)

        conv43 = self.upsample2(conv43)
        conv432 = torch.cat((self.upsample2(conv4_feat), conv43, conv2_feat), 1)
        conv432 = self.racb_432_2(conv432)

        conv432 = self.conv432_2(conv432)

        conv432 = self.upsample2(conv432)
        conv4321 = torch.cat((self.upsample4(conv4_feat), self.upsample2(conv43), conv432, conv1_feat), 1)
        conv4321 = self.racb_4321(conv4321)
        pred_ref = self.layer7(conv4321)


        return self.upsample8(pred_init), self.upsample4(pred_ref)

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)


class Saliency_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self):
        super(Saliency_feat_encoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.resnet = B2_ResNet()
        if self.training:
            self.initialize_weights()

        # self.conv_edge_1 = nn.Conv2d(64, 32, 1)
        # self.conv_edge_2 = nn.Conv2d(512, 32, 1)
        # self.conv_edge_4 = nn.Conv2d(2048, 32, 1)
        # self.conv_edge_11 = nn.Conv2d(32, 32, 3, padding=1)
        # self.conv_edge_21 = nn.Conv2d(512, 32, 1)
        # self.conv_edge_41 = nn.Conv2d(2048, 32, 1)


    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x0 = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x0)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

        return x1,x2,x3,x4

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

class _ConvBatchNormReLU(nn.Sequential):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(out_channels),
        )

        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)

class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, output_stride):
        super(_ASPPModule, self).__init__()
        if output_stride == 8:
            pyramids = [12, 24, 36]
        elif output_stride == 16:
            pyramids = [6, 12, 18]
        self.stages = nn.Module()
        self.stages.add_module(
            "c0", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d((1,1))),
                    ("conv", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)),
                ]
            )
        )
        self.fire = nn.Sequential(
            OrderedDict(
                [
                    ("conv", _ConvBatchNormReLU(out_channels * 5, out_channels, 3, 1, 1, 1)),
                    ("dropout", nn.Dropout2d(0.1))
                ]
            )
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        h = self.fire(h)
        return h

class Cod_feat_encoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self):
        super(Cod_feat_encoder, self).__init__()
        self.resnet = B2_ResNet()
        self.aspp = _ASPPModule(2048, 256, 16)
        if self.training:
            self.initialize_weights()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x0 = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x0)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

        return x1, x2, x3, x4

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)



class Compute_latent_feature(nn.Module):
    def __init__(self, channel, latent_size):
        super(Compute_latent_feature, self).__init__()
        self.conv1 = Triple_Conv(256, channel)
        self.conv2 = Triple_Conv(512, channel)
        self.conv3 = Triple_Conv(1024, channel)
        self.conv4 = Triple_Conv(2048, channel)
        self.fc1 = nn.Linear(channel * 4 * 11 * 11, latent_size)
        self.upsample05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample025 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.upsample0125 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.channel = channel

    def forward(self, x1,x2,x3,x4):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        output = torch.cat((self.upsample0125(x1), self.upsample025(x2), self.upsample05(x3), x4), 1)
        print(output.shape)
        output = output.view(-1, self.channel * 4 * 11 * 11)
        print(output.shape)
        output = self.fc1(output)
        return output


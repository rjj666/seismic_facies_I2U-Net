import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class eca_layer(nn.Module):
    """
    Efficient Channel Attention layer
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

def BNReLU(num_features):
    """
    批量归一化和ReLU激活函数的组合
    """
    return nn.Sequential(
        nn.BatchNorm2d(num_features),
        nn.ReLU()
    )

class SPP_inception_block(nn.Module):
    """
    空间金字塔池化Inception模块
    """
    def __init__(self, in_channels):
        super(SPP_inception_block, self).__init__()
        # 使用自适应池化以适应不同尺寸的输入
        self.pool1 = nn.AdaptiveMaxPool2d(output_size=(2, 2))
        self.pool2 = nn.AdaptiveMaxPool2d(output_size=(3, 3))
        self.pool3 = nn.AdaptiveMaxPool2d(output_size=(5, 5))
        self.pool4 = nn.AdaptiveMaxPool2d(output_size=(6, 6))

        self.dilate1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, padding=0)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        b, c, h, w = x.size()
        # 使用自适应池化以适应不同尺寸的输入
        pool_1 = self.pool1(x).view(b, c, -1)
        pool_2 = self.pool2(x).view(b, c, -1)
        pool_3 = self.pool3(x).view(b, c, -1)
        pool_4 = self.pool4(x).view(b, c, -1)
        
        pool_cat = torch.cat([pool_1, pool_2, pool_3, pool_4], -1)
        
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))

        cnn_out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        cnn_out = cnn_out.view(b, c, -1)
        
        out = torch.cat([pool_cat, cnn_out], -1)
        out = out.permute(0, 2, 1)
        
        return out

class NonLocal_spp_inception_block(nn.Module):
    """
    非局部SPP Inception模块
    """
    def __init__(self, in_channels=512, ratio=2):
        super(NonLocal_spp_inception_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.key_channels = in_channels//ratio
        self.value_channels = in_channels//ratio
            
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0),
            BNReLU(self.key_channels),
        )
                                  
        self.f_query = self.f_key
        
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                kernel_size=1, stride=1, padding=0)
                                 
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                          kernel_size=1, stride=1, padding=0)

        self.spp_inception_v = SPP_inception_block(self.key_channels)
        self.spp_inception_k = SPP_inception_block(self.key_channels)
        
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        
        x_v = self.f_value(x)
        value = self.spp_inception_v(x_v)
        
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        
        x_k = self.f_key(x)
        key = self.spp_inception_k(x_k)
        key = key.permute(0, 2, 1)
        
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        
        return context

class MFII_DecoderBlock(nn.Module):
    """
    多特征交互集成解码器块
    """
    def __init__(self, in_channels, n_filters, rla_channel=32, ECA_size=5):
        super(MFII_DecoderBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

        self.eca = eca_layer(n_filters, k_size=ECA_size)

    def forward(self, x, skip=None):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        
        if skip is not None:
            if x.size() != skip.size():
                x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)
            x = x + skip
            
        x = self.relu3(x)
        x = self.eca(x)
        
        return x

class Section_I2U_Net(nn.Module):
    """
    改进的I2U-Net，适用于任意尺寸输入，特别是地震剖面数据
    """
    def __init__(self, classes=6, channels=1):
        super(Section_I2U_Net, self).__init__()
        
        filters = [64, 128, 256, 512]
        # 更新为新的API，避免deprecated警告
        try:
            # 尝试使用新API
            from torchvision.models import ResNet34_Weights
            resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=ResNet34_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            # 如果是旧版本PyTorch，回退到旧API
            resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        
        # Modify first conv layer to accept single channel
        if channels == 1:
            self.firstconv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.firstconv.weight.data = torch.sum(resnet.conv1.weight.data, dim=1, keepdim=True)
        else:
            self.firstconv = resnet.conv1
            
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # HIFA modules
        self.hifa1 = NonLocal_spp_inception_block(filters[0], ratio=2)
        self.hifa2 = NonLocal_spp_inception_block(filters[1], ratio=2)
        self.hifa3 = NonLocal_spp_inception_block(filters[2], ratio=2)
        self.hifa4 = NonLocal_spp_inception_block(filters[3], ratio=2)

        # Decoder path
        self.decoder4 = MFII_DecoderBlock(filters[3], filters[2])
        self.decoder3 = MFII_DecoderBlock(filters[2], filters[1])
        self.decoder2 = MFII_DecoderBlock(filters[1], filters[0])
        self.decoder1 = MFII_DecoderBlock(filters[0], filters[0])

        # Final classification layers
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, classes, 3, padding=1)

    def forward(self, x):
        # 保存输入尺寸，用于最终输出尺寸的调整
        input_size = x.size()
        
        # Encoder path
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)
        
        e1 = self.encoder1(x_)
        e1 = e1 + self.hifa1(e1)
        
        e2 = self.encoder2(e1)
        e2 = e2 + self.hifa2(e2)
        
        e3 = self.encoder3(e2)
        e3 = e3 + self.hifa3(e3)
        
        e4 = self.encoder4(e3)
        e4 = e4 + self.hifa4(e4)

        # Decoder path with skip connections
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, x)

        # Final classification
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        
        # 确保输出尺寸与输入匹配（除了通道数）
        if out.size(2) != input_size[2] or out.size(3) != input_size[3]:
            out = F.interpolate(out, size=(input_size[2], input_size[3]), 
                               mode='bilinear', align_corners=False)

        return out
    
    def get_prob(self, outputs):
        """
        将网络输出转换为概率
        Args:
            outputs: 网络输出张量
        Returns:
            概率张量
        """
        # 对通道维度应用softmax
        return F.softmax(outputs, dim=1)
    
    def get_attention(self):
        """
        获取注意力图（如果有的话）
        Returns:
            注意力图张量或None
        """
        # 这里可以返回模型中的注意力图
        # 如果模型没有注意力机制，可以返回None
        return None 
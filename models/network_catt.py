import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import ResidualBlockNoBN, default_init_weights, make_layer


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        k = int(abs((log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1)
        y = y.unsqueeze(1)
        y = self.conv(y)
        return y.squeeze(2).squeeze(1)

class Attention(nn.Module):
    def __init__(self, channel):
        super(Attention, self).__init__()
        #self.scale = 1. / (channel ** 0.5)
    
    def forward(self, query, key, value):
        # 元素相乘
        energy = query * key
        #print('self.scale:', self.scale)
        # 计算注意力权重，这里假设您希望将元素相乘的结果直接用作注意力权重
        attention = F.softmax(energy, dim=-1)
        
        #print('attention:', attention.shape)
        #print('value:', value.shape)
        # 将attention形状从[32, 64]扩展为[32, 64, 1, 1]以匹配value的空间维度
        attention_expanded = attention.unsqueeze(-1).unsqueeze(-1)
        # 应用注意力权重到value
        # 注意这里需要调整attention的形状以匹配value的形状，或者重新设计机制
        # 此处代码作为示例，可能需要根据您的具体需求进行调整
        # 假设我们简单地通过扩展dim=1然后进行广播来模拟这一过程
        out = attention_expanded * value
        #print('out:',out.shape)
        return out


class CAtt(nn.Module):
    def __init__(self, num_in_ch=6, num_out_ch=3, num_feat=64, num_blocks=6):
        super(CAtt, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch // 2, num_feat, 1, 1, 0)
        # 为MAE特征处理创建独立的ECA模块
        self.conv_mae = nn.Conv2d(num_in_ch // 2, num_feat, 1, 1, 0)
        self.eca_mae = ECA(num_feat)
        self.attention = Attention(num_feat)

        # 使用ModuleList来保存每个block和对应的ECA
        self.body = nn.ModuleList()
        for _ in range(num_blocks):
            # 创建残差块组合
            res_blocks = make_layer(ResidualBlockNoBN, num_basic_block=2, num_feat=num_feat)
            # 创建对应的ECA模块
            eca_block = ECA(num_feat)
            # 将残差块组合和ECA模块封装成一个nn.Sequential对象
            block_with_eca = nn.Sequential(res_blocks, eca_block)
            # 添加到ModuleList中
            self.body.append(block_with_eca)

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(weights_init_kaiming)

    def forward(self, x):
        feat_x = x[:, :3, :, :]
        mae_mem = x[:, 3:6, :, :]
        #print('1')
        #print(feat_x.shape)
        #print(mae_mem.shape)
        feat_x = self.lrelu(self.conv_first(feat_x))
        mae_mem = self.lrelu(self.conv_mae(mae_mem))
        key = self.eca_mae(mae_mem)
        #print(key.shape)
        #print('start')
        #print(feat_x.shape)
        #print(mae_mem.shape)
        for res_blocks, eca_block in self.body:
            #print('body')
            feat_x = res_blocks(feat_x)
            #print(feat_x.shape)
            query = eca_block(feat_x)
            #print(query.shape)
            feat_x = self.attention(query, key, mae_mem) + feat_x

        out = self.conv_last(feat_x)
        return out
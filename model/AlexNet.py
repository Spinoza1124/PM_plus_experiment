import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from base import BaseModel

def init_layer(layer):
    """初始化线性层或者卷积层"""
    nn.init.xavier_uniform_(layer.weight)
    
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """初始化BN层"""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(0.)

class SER_AlexNet(BaseModel):
    """
    Reference:
    https://pytorch.org/docs/stable/torchvision/models.html#id1

    AlexNet model from torchvision package. The model architecture is slightly
    different from the original model.
    See: AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Parameters
    ----------
    num_classes : int
    in_ch   : int
        The number of input channel.
        Default AlexNet input channels is 3. Set this parameters for different
            numbers of input channels.
    pretrained  : bool
        To initialize the weight of AlexNet.
        Set to 'True' for AlexNet pre-trained weights.

    Input
    -----
    Input dimension (N,C,H,W)

    N   : batch size
    C   : channels
    H   : Height
    W   : Width

    Output
    ------
    logits (before Softmax)
    """
    def __init__(self, num_classes=4, in_ch=3, pretrained=True):
        super(SER_AlexNet, self).__init__()
        
        model = torchvision.models.alexnet(pretrained)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        
        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            init_layer(self.features[0])
        
        self.classifier[6] = nn.Linear(4096, num_classes)
        
        self._init_weights(pretrained=pretrained)
        
        print('\n<< SER AlexNet Finetuning model initialized >>\n')
    
    def forward(self, x):
        # self.features 包含了 AlexNet 的所有卷积层、池化层、激活函数等
        # 将输入数据通过 AlexNet 的特征提取层（卷积层部分）
        x = self.features(x)
        
        # 通过自适应平均池化层处理特征图
        # self.avgpool 是 nn.AdaptiveAvgPool2d((6, 6))
        # 将特征图的空间维度统一调整为 6×6
        # 输出形状变为 (N, 256, 6, 6)
        x = self.avgpool(x)
        # 将多维张量展平为二维张量
        # 参数 1 表示从第1个维度开始展平（保持batch维度不变）
        # 将 (N, 256, 6, 6) 展平为 (N, 256*6*6) = (N, 9216)
        # 这是为了准备输入到全连接层
        x = torch.flatten(x, 1)
        # 通过分类器（全连接层）进行最终分类
        # self.classifier 包含多个线性层和dropout层
        # 最后一层输出维度是 num_classes（您设置的类别数）
        # out 的形状是 (N, num_classes)，包含每个类别的预测分数（logits）
        out = self.classifier(x)
        
        # 返回两个值：
        # x: 展平后的特征向量 (N, 9216)，可用于特征分析
        # out: 最终的分类输出 (N, num_classes)，用于计算损失和预测
        return x, out
    
    def _init_weights(self, pretrained=True):
        init_layer(self.classifier[6])
        
        if pretrained == False:
            init_layer(self.features[0])
            init_layer(self.features[3])
            init_layer(self.features[6])
            init_layer(self.features[8])
            init_layer(self.features[10])
            init_layer(self.classifier[1])
            init_layer(self.classifier[4])
            
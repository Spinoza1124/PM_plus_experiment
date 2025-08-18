import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """
    训练数据集类
    
    Parameters
    ----------
    data : dict
        包含输入数据的字典，包括:
        - seg_spec: 频谱特征 (N, C, H, W)
        - seg_mfcc: MFCC特征
        - seg_audio: 音频特征
        - seg_label: 标签 (N,)
    num_classes : int
        类别数量，默认为4 
    """
    def __init__(self, data, num_classes=4):
        super(TrainDataset, self).__init__()
        self.data_spec = data['seg_spec']
        self.data_mfcc = data['seg_mfcc']
        self.data_audio = data['seg_audio']
        self.seg_label = data['seg_label']
        self.n_samples = len(self.seg_label)
        self.num_classes = num_classes
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        sample = {
            'seg_spec': self.data_spec[index],
            'seg_mfcc': self.data_mfcc[index],
            'seg_audio': self.data_audio[index],
            'seg_label': self.seg_label[index]
        }
        return sample
    
    def get_preds(self, preds):
        """
        这段代码是一个预测后处理函数，用于将模型的原始输出转换为最终的预测标签
        """
        preds = np.argmax(preds, axis=1)
        return preds
    
    def weighted_accuracy(self, predictions):
        """
        计算预测结构的准确率分数
        predictions: ndarray
            模型的预测结果
        
        Returns
        -------
        float
            Accuracy score.
        """
        acc = (self.seg_label == predictions).sum() / self.n_samples
        return acc
    
    def unweighted
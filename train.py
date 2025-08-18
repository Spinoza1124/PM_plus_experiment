from utils import ConfigParser
import torch
import argparse
import numpy as np
import collections
import data_loader.data_loaders as module_data
import model.model as module_arch
from trainer import Trainer
import model.loss as module_loss
import model.metric as module_metric

# 固定随机数种子
SEED = 123
torch.manual_seed(SEED)
# 将cuDNN的确定性算法开关打开。为了追求极致的性能，cuDNN的某些算法本身是非确定性的。将这个标志位设置为True,就是获得确定性的算法
torch.backends.cudnn.deterministic = True
# 关闭cuDNN的自动基准测试功能。选项为True时，cuDNN会在程序开始运行时，针对当前的模型的网络结构，测试不同的算法，然后选择一个最快的来执行。这个寻找最快的算法过程本身也可能带来不确定性。
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# 这个函数是整个程序的总指挥。它不负责具体的训练细节，而是负责根据 config 对象，组装出所有需要的部件（数据、模型、优化器等），然后把它们交给 Trainer 去执行。
def main(config: ConfigParser) -> None:
    # 从 config 对象中获取一个名为 'train' 的日志记录器。
    # 使用日志系统 (logging) 而不是 print() 是专业做法。它可以方便地将信息输出到控制台和文件，并且可以分级（INFO, DEBUG, WARNING），便于管理。
    logger = config.get_logger('train')

    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    
    # 构造网络模型，然后在控制台打印
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    if 'n_gpu' in config.config:
        gpu_setting = config.config['n_gpu']
        if isinstance(gpu_setting, list):
            if len(gpu_setting) > 0:
                device = torch.device(f'cuda:{gpu_setting[0]}' if torch.cuda.is_available() else 'cpu')
                device_ids = gpu_setting
            else:
                device = torch.device('cpu')
                device_ids = []
        else:
            device = torch.device(f'cuda:{gpu_setting}' if torch.cuda.is_available() else 'cpu')
            device_ids = [gpu_setting] if gpu_setting >= 0 else []
    else:
        device = torch.device('cpu')
        device_ids = []
    
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    
    trainer.train()
    

if __name__ == "__main__":
    # 创建一个解析器对象，把命令行里敲的“字符串”变成 Python 可以直接拿来用的“变量”
    args = argparse.ArgumentParser(description="协同注意力语音情感识别训练脚本")
    # 用来指定配置文件的路径
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    # 用来指定一个之前保存的检查点 (checkpoint) 文件的路径，以便从上次中断的地方继续训练。
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    # 用来指定使用哪些 GPU
    args.add_argument('-d', '--device', default=None, type=str, help="indices of GPUs to enable (default: all)")
    # 使用自定义 CLI 选项，根据 YAML 文件中给定的默认值修改配置。这里创建了一个命名元组 (namedtuple)。你可以把它看作一个轻量级的、只有属性没有方法的类。创建一个 CustomArgs 对象会像这样：CustomArgs(flags=[...], type=..., target=...)。
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
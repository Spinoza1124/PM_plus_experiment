from __future__ import annotations
from argparse import ArgumentParser
import os
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from logger import setup_logging
from train import CustomArgs
from utils import read_yaml, write_yaml
from functools import reduce, partial
from operator import getitem
from typing import Dict, Optional, List, Any


class ConfigParser:
    def __init__(self, config: Dict[str, Any], resume: Optional[Path] = None, modification=None, run_id=None) -> None:
        """
        解析配置文件的类。处理训练、初始化，检查点保存和日志
        :param config: 配置文件
        :param resume: String,checkpoint被加载的路径
        :param modification: Dict keychain:value, 可以替换的配置文件的值
        :param run_id: 唯一的训练过程的标识符。用于保存checkpoints和训练日志。
        """
        # 这一行接收原始的 config 字典和 modification 字典，然后调用一个辅助函数 _update_config 来将 modification 中的更改应用到 config 中。结果存储在 self._config 这个内部变量里。
        # 是实现“命令行覆盖配置文件”功能的核心。它确保了最终生效的配置是基础配置和命令行自定义配置的结合体。使用 _config (带下划线) 是一种约定，表示这是一个内部变量，不应该被外部直接修改。
        self._config = _update_config(config, modification)
        # 存储 resume 参数，它是一个指向检查点 (checkpoint) 文件的路径
        self.resume = resume

        # 从配置中读取顶层的保存目录（例如 "saved/"），并使用 pathlib.Path 将其转换为一个路径对象。
        # pathlib 是现代 Python 中处理文件路径的最佳实践。它能跨平台（Windows/Linux/macOS）无缝工作，并且允许使用 / 操作符来拼接路径，比传统的字符串拼接更安全、更清晰。
        save_dir = Path(self.config['trainer']['save_dir'])
        
        # 从配置中获取实验的名称，例如 "MNIST_LeNet".
        #  用于创建有意义的文件夹名称，方便日后查找和区分不同的实验。
        exper_name = self.config['name']
        # 检查是否传入了 run_id。如果没有，就使用当前的日期和时间（格式如 0816_102030）作为默认的 run_id。
        if run_id is None:
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        # 使用 pathlib 的 / 操作符构建最终的模型保存路径和日志保存路径。例如 saved/models/MNIST_LeNet/0816_102030/。
        # 建立一个清晰、结构化的目录树。将模型和日志分开存放，并按实验名称和运行ID进行组织，是管理大量实验的最佳实践。
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # 创建上面定义的模型保存目录和日志目录。
        # 程序需要这些文件夹来存放文件。parents=True 选项会自动创建所有必需的父目录（类似 mkdir -p）。exist_ok=True 保证了如果目录已经存在，程序不会报错（这在恢复训练时很有用）。
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # 将最终生效的（已经被命令行修改过的）配置字典保存为一个 JSON 文件，放在本次运行的模型保存目录中。
        # 这是保证可复现性的关键一步！ 当你几个月后回头看一个训练好的模型时，这个 config.yaml 文件会告诉你当时所使用的全部超参数，包括任何临时的命令行修改。
        write_yaml(self.config, self.save_dir / 'config.yaml')

        # 调用外部函数来配置 Python 的 logging 模块，使其将日志保存到 self.log_dir 中。同时定义一个字典，将简单的数字（0, 1, 2）映射到标准的日志级别。
        #  集中管理日志配置。log_levels 字典提供了一个用户友好的接口，可以通过一个简单的数字来控制日志的详细程度（verbosity）。
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    # 这是一个类方法工厂。它的作用不是操作一个已有的对象，而是根据命令行参数创建并返回一个全新的 ConfigParser 对象。
    @classmethod
    def from_args(cls, parser: ArgumentParser, options: List[CustomArgs]=[]) -> ConfigParser:
        """
        从cli 参数初始化类。用于 train 和 test
        """
        # 遍历 options 列表（包含了 --lr, --bs 等自定义选项），并将它们添加到 argparse 解析器中。
        # 动态地将所有自定义的命令行选项注册到 argparse，使其能够被识别和解析。
        for opt in options:
            parser.add_argument(*opt.flags, default=None, type=opt.type)

        # 解析器对象 变成 解析结果对象.args 不再是“解析器”，而是“解析后的参数集合”，可以直接用 args.lr、args.batch_size 等字段。
        args = parser.parse_args()
        
        # 如果用户通过 --device 指定了 GPU，就设置 CUDA_VISIBLE_DEVICES 环境变量。
        # 这是在 PyTorch 等框架中控制程序能“看到”哪些 GPU 的标准方法。在程序导入 torch 之前设置此变量，可以有效隔离 GPU。
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        
        # 处理恢复训练的情况。它将 cfg_fname (配置文件名) 指向被恢复的检查点所在目录的 config.json。
        # 保证在恢复训练时，使用的是原始实验的配置，而不是一个可能不匹配的新配置文件，从而确保一致性。
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = Path(args.config)
        else:
            #  处理开始新训练的情况。它断言（assert）用户必须通过 -c 提供一个配置文件，否则程序会报错并退出。
            # 保证程序总是有配置可以遵循。
            msg_no_cfg = "Configuration file need to be specified.Add '-c config.yaml', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        # 读取找到的配置文件。
        config = read_yaml(cfg_fname)

        # 这是一个用于微调 (fine-tuning) 的高级功能。如果用户既提供了 --resume 又提供了 --config，它会先加载旧的配置，然后用新配置文件中的内容去更新它。
        if args.config and resume:
            # 更新新的配置文件
            config.update(read_yaml(args.config))

        # 一个字典推导式。它遍历 options，从解析后的 args 中获取每个自定义选项的值（例如，获取 --lr 后面跟的 0.01），然后创建一个形如 {'optimizer;args;lr': 0.01} 的字典。
        # 这就是构建 modification 字典的地方，它将命令行输入与配置字典中的目标路径精确地关联起来。
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)
    
    # 这是一个“对象工厂”。
    # 它在配置字典中查找 name 对应的部分（例如 config['optimizer']），读取 type 字段（例如 "Adam"）作为类名，读取 args 字段作为初始化参数，然后从给定的 module（例如 torch.optim）中找到这个类并创建它的实例。
    # 你的训练代码不需要写 torch.optim.Adam(...)，而是写 config.init_obj('optimizer', torch.optim)。这样，如果你想换成 SGD 优化器，只需要修改 YAML 文件，完全不需要改动 Python 代码。这极大地提升了实验的灵活性。
    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)
    
    # 这是一个“函数工厂”。
    # 和 init_obj 非常相似，但它不直接创建和返回对象实例，而是使用 functools.partial 返回一个已经配置好参数的函数。
    # 当你需要一个函数而不是一个对象时非常有用。例如，你可以配置一个特定的损失函数（带有特定的 weight 或 reduction 参数），init_ftn 会返回这个配置好的损失函数，你可以在之后直接调用它。
    def init_ftn(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)
    
    # 实现了字典风格的访问，即 config['trainer']。
    # 提供一种方便、直观的方式来访问配置项，就像操作普通的 Python 字典一样。
    def __getitem__(self, name):
        return self.config[name]
    
    #  获取一个配置好的 logger 实例。
    # 封装了获取 logger 的逻辑，并集成了日志详细程度的控制。
    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # 设置只读属性
    # 将 _config, _save_dir, _log_dir 这些内部变量暴露为只读属性。
    # 这是一种保护机制。外部代码可以通过 config.config 来读取配置，但不能通过 config.config = new_dict 来意外地修改它，增强了类的封装性和健壮性。
    @property
    def config(self):
        return self._config
    
    @property
    def save_dir(self):
        return self._save_dir
    
    @property
    def log_dir(self):
        return self._log_dir

# 遍历 modification 字典，对于每一个键值对，调用 _set_by_path 来更新 config 字典。
# 这是应用所有修改的入口点。
def _update_config(config, modification):
    if modification is None:
        return config
    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

# 从一个标志列表（如 ['--lr', '--learning_rate']）中提取出一个规范的名称（'lr' 或 'learning_rate'）。
# argparse 会将命令行参数存储为对象的属性，这个函数就是用来获取那个属性名。
def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

# 核心魔法之一。它接收一个字典 (tree)，一个用分号分隔的路径字符串 (keys) 和一个值。它会沿着路径找到目标位置并设置新值。
# 这是实现 target='optimizer;args;lr' 这种深层路径修改的关键。它通过调用 _get_by_path 来首先定位到目标位置的父字典。
def _set_by_path(tree: Dict[str, Any], keys: str, value: Any) -> None:
    key_list = keys.split(';')
    _get_by_path(tree, key_list[:-1])[keys[-1]] = value

# 另一个核心魔法。它使用 functools.reduce 和 operator.getitem 来沿着 keys 列表在嵌套字典中“向下走”。
# reduce(getitem, ['optimizer', 'args'], config) 等价于 config['optimizer']['args']。这是一种非常简洁和函数式的编程方式，用于访问深层嵌套的字典元素。
def _get_by_path(tree: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    return reduce(getitem, keys, tree)
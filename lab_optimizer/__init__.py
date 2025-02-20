"""
lab_optimizer

author : Zifeng Li
email : 221503020@smail.nju.edu.cn
github_url : https://github.com/quantumopticss/lab_optimizer

=========

Provides
    1. global_optimizer
    2. local_optimizer
    3. mloop_optimizer
    4. torch_optimizer (only for torch functions)
    5. powerful visualization tools
    6. physics constants and units conversion 
    7. lab_optimizer examples
    8. flexible custom opt algorithm API and visualization API

MIT License

Copyright (c) 2025 Zifeng Li
All rights reserved.

part of this project are published under other open source licenses, refer to the License profile

"""

__version__ = "1.2.4"
__all__ = ["local_optimize","global_optimize","mloop_optimize","torch_optimize",
           "log_visiual","local_time","read_log",
           "examples","units"]

import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .local_optimize import local_optimize
from .mloop_optimize import mloop_optimize
from .torch_optimize import torch_optimize
from .global_optimize import global_optimize
from .optimize_base import log_visiual, local_time, read_log
from .opt_examples import examples
from . import units

del os, sys
def opt_random_seed(np_seed:int = 37, th_seed:int = 37):
    """set random seeds for numpy and torch module to ensure the repeatability of experiments

    Args
    ---------
    np_seed : int
        Random seed for NumPy, default is 37.
    th_seed : int
        Random seed for PyTorch, default is 37.

    Notes:
    - When setting random seeds, ensure that the libraries used support setting random seeds. For example, both NumPy and PyTorch support setting random seeds, but other libraries may not.
    - When setting random seeds, ensure that the seed value is fixed. If a different seed value is used each time the program is run, the experimental results will not be repeatable.
    - When setting random seeds, note that the random seed settings of different libraries may affect each other. For example, setting the random seed for PyTorch may affect the random seed for NumPy, and vice versa. Therefore, when setting random seeds, consider the needs of different libraries comprehensively.
    """
    from torch import manual_seed
    from torch.cuda import manual_seed_all
    from numpy.random import seed
    
    manual_seed(th_seed)
    manual_seed_all(th_seed)
    seed(np_seed)

__version__ = "1.2.0"

"""
lab_optimizer
=========
URL : https://github.com/quantumopticss/lab_optimizer

Provides
  1. global_optimizer
  2. local_optimizer
  3. mloop_optimizer
  4. torch_optimizer (only for torch functions)
  5. powerful visualization tools
  6. physics constants and units conversion 
  7. lab_optimizer examples

MIT License

Copyright (c) 2025 Zifeng Li
email : 221503020@smail.nju.edu.cn

All rights reserved.

part of this project are published under other open source licenses, refer to the License profile

"""

import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .local_optimize import local_optimize
from .mloop_optimize import mloop_optimize
from .torch_optimize import torch_optimize
from .global_optimize import global_optimize
from .optimize_base import log_visiual, local_time
from .opt_examples import examples
from . import units

del os, sys

__version__ = "1.1.9"

"""
lab_optimizer
=========

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

part of this project are published under 

BSD 3-Clause License : <https://github.com/scikit-learn/scikit-learn>, <https://github.com/scipy/scipy>, <https://github.com/pandas-dev/pandas>, <https://github.com/mwaskom/seaborn>
MIT License : <https://github.com/michaelhush/M-LOOP>, <https://github.com/guofei9987/scikit-opt>, <https://github.com/plotly/plotly.py>
Other License : <https://github.com/pytorch/pytorch>, <https://github.com/numpy/numpy>,<https://github.com/matplotlib/matplotlib>

"""

import os,sys
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))

from .local_optimize import local_optimize
from .mloop_optimize import mloop_optimize
from .torch_optimize import torch_optimize
from .global_optimize import global_optimize
from .optimize_base import *
from .opt_examples import examples
from . import units
    
def _main():
    import numpy as np
    def func(x,a,b,c,d):
        vec = np.array([a,b,c,d])
        f = np.sum((x - vec)**2,axis = None) + 5*np.sum(np.cos(x-a) + np.cos(x-b) + np.sin(x-c) + np.sin(x-d)) + a*b*c*d
        uncer = 0.1
        bad = None
        return_dict = {'cost':f,'uncer':uncer,'bad':bad}
        return return_dict
    
    init = np.array([3,0,4,2])
    a = 6
    b = 8
    c = 1
    d = 2
    bounds = ((-10,10),(-10,10),(-10,10),(-10,10))
    method_list =  ["dual","simplex"]
    optimizer_list = [global_optimize,local_optimize]
    multi_optimize(func,init,args = (a,b,c,d,),optimizer_list=optimizer_list,bounds_list = [bounds],
                   max_run_list = [1,20],delay = 0.03,method_list = method_list,extra_dict_list=[{},{}],val_only = True,msg = False,log = True)
     
if __name__ == "__main__":
    _main()
    
del _main
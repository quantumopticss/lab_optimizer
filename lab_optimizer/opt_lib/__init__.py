__global__ = ["ISMA"]
__local__ = [""]

"this libs aim at providing extensions for lab_optimizer"
# you can add your own optimization algorithm here, with a fix interface 

# ** assume that _alg is your opt algorithm **
# res = _alg(func,paras_init,bounds,,args,*,**self._extra_dict)

# x_opt = res.run() 
# OR OR OR 
# res.run(), x_opt = res.x

# example:
class test_alg:
    def __init__(self,func,paras_init,bounds,args = (),**kwargs):
        self.x = paras_init
            
    def run(self):
        import numpy as np
        for i in range(10):
            self.x = self.x + np.random.randn(*self.x.shape)*1e-3
            print(self.x)
            
del test_alg

import importlib
from .ISMA import ISMA

def get_method(func_name:str,module_name:str = None) -> callable:
    """return optimization algorithm corresponds to name

    Args
    ---------
    name : str
        name of algorithms
        
    module_name : str
        name of to import modules, defeault is None (this module : opt_lib)
    """
    if module_name:
        try:
            module = importlib.import_module(f"{__package__}.{module_name}")
            return getattr(module,func_name,None)
        except:
            return None
    else:
        func = globals().get(func_name, None)
    
    return func

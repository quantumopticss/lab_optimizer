__optlib_global__ = ["ISMA"]
"""global optimization algorithms in opt_lib"""

__optlib_local__ = [""]
"""lobal optimization algorithms in opt_lib"""

# this libs aim at providing extensions for lab_optimizer
# you can add your own optimization algorithm here, with a fix interface 

# to add custom defined algorithm, you need to : 
# 1. follow the general parameters name (you can require some parameters in **extra_dict if necessary)
# 2. provide general interface XXX.run() , XXX.x_optimize 

### example:
# ** assume that _alg is your opt algorithm **
# res = _alg(func,paras_init,bounds,args,**extra_dict)
# x_optimize = res.run()

class test_alg:
    def __init__(self,func,paras_init,bounds,args = (),**kwargs):
        self._x = paras_init
        self._kwargs = kwargs
    def run(self):
        ## operate
        from numpy.random import rand
        for i in range(self._kwargs.get("max_run",10)):
            self._x += rand(self._x.shape)
            
        ## finish
        self.x_optimize = self._x
        return self.x_optimize
            
del test_alg

class LIB_ERROR(Exception):
    def __init__(self,opt):
        error_dict = dict(
            not_found = "algorithm not found, or invalid name",
        )
        if opt in error_dict:
            msg = error_dict[opt]
        else:
            msg = opt 
        raise Exception("opt_lib error : " + msg)

def get_method(func_name:str) -> callable:
    """return optimization algorithm corresponds to name

    Args
    ---------
    name : str
        name of algorithms, should be either in __optlib_local__ or in __optlib_global__
    """
        
    if func_name in __optlib_local__ or func_name in __optlib_global__:
        exec(f"from {func_name} import {func_name}")
        return eval(func_name)
    else:
        print("optlib for local : " + str(__optlib_local__))
        print("optlib for global : " + str(__optlib_global__))
        LIB_ERROR("not_found")
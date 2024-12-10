import os,sys
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))

from .local_optimize import local_optimize
from .mloop_optimize import mloop_optimize
from .torch_optimize import torch_optimize
from .global_optimize import global_optimize
from .optimize_base import *
from .units import *

"""
``func`` should be a callable match the optimizer
and func should return a dict {"cost":cost,"uncer":uncer,"bad":bad}

cost : float -> cost value
uncer : float -> uncertainty of the cost
bad : bool -> whether the run is bad (bad = True represent bad run)

you are suggested to calculate uncer and bad, but they are required only in mloop_optimize,
and uncer and bad will not be used in other optimizers 

call th optimization algorithm ``XXX_optimize``

    `` XXX_optimize(func,paras_init,args,bounds,extra_dict,kwargs) ``

Args
--------
fun : callable
    The objective function to be minimized.

        ``fun(x, *args) -> dict : {'cost':float, 'uncer':float, 'bad':bool}``
        
    where ``cost`` is the value to minimize, ``uncer`` is uncertainty,
    ``bad`` is the judge whether this value is bad (bad = True) for this cost

    ``x`` is a 1-D array with shape (n,) and ``args``
    is a tuple of the fixed parameters needed to completely
    specify the function.

paras_init : ndarray, shape (n,)
    Initial guess. Array of real elements of size (n,),
    where ``n`` is the number of independent variables.

args : tuple, optional
    Extra arguments passed to the objective function which will not
    change during optimization
    
bounds : sequence or `Bounds`, optional
    Bounds on variables
        should be Sequence of ``(min, max)`` pairs for each element in `x`. None is used to specify no bound.

kwArgs
---------
method : string
    optimization algorithm to use 

extra_dict : dict
    used to transfer specific arguments for optimization algorithm
    
opt_inherit : class 
    inherit ``optimization results``, ``parameters`` and ``logs``
    defeault is None (not use inherit) 

delay : float 
    delay of each iteration, default is 0.1s

max_run : int 
    maxmun times of running optimization, default = 10 

msg : Bool
    whether to output massages in every iterarion, default is True
    
log : Bool
    whether to generate a log file in labopt_logs
    
logfile : str
    log file name , defeault is "optimization__ + <timestamp>__ + <method>__.txt"

"""
    
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
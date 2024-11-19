import mloop.interfaces as mli
import mloop.controllers as mlc
# import mloop.visualizations as mlv
from .optimize_base import optimize_base
import numpy as np

class _mloops_interface(mli.Interface,optimize_base):
    def __init__(self,func,args):
        mli.Interface.__init__(self)
        self._func1 = func
        self._args1 = args
    
    #the method that runs the experiment given a set of parameters and returns a cost
    def get_next_cost_dict(self,params_dict):

        #The parameters come in a dictionary and are provided in a numpy array
        params = params_dict['params']
        cost_dict = self._func1(params,*self._args1)
        
        #The cost, uncertainty and bad boolean must all be returned as a dictionary
        return cost_dict
    
class mloop_optimize(optimize_base):
    """reconstructed mloop a good integrated lab used optimization algorithm, ``methods`` including:
        - ``'gaussian_process', 'neural_net', 'differential_evolution',  'simplex', 'random'``

        Warning
        ---------
        in mloop_optimize, the func is required to return a valid uncer and bad and val_only is automatically set to False  
        
        Args
        ---------
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
        
        kwArgs
        ---------
        extra_dict : dict
            used for extra parameters for mloop controller
        
        target : float
            target cost of optimization function, defeault is  -infty
        
        method : string 
            method of scipy.optimize.minimize to be used, 
            should be one of: 'gaussian_process', 'neural_net', 'differential_evolution',  'simeplx' which is "nelder_mead", 'random'
            
        bounds : sequence or `Bounds`, optional
            Bounds on variables

                Should be Sequence of ``(min, max)`` pairs for each element in `x`. None is used to specify no bound.
        
        delay : float 
            delay of each iteration, default is 0.1s
        
        max_run : int 
            maxmun times of running optimization, default = 100
        
        msg : Bool
            whether to output massages in every iterarion, default is True
            
        log : Bool
            whether to generate a log file in labopt_logs
            
    """
    def __init__(self,func,paras_init,args = (),extra_dict = {},bounds = None,**kwargs):
        kwargs["val_only"] = False # let f return cost dict instead of a cost value
        kwargs["msg"] = None # use mloop msg
        optimize_base.__init__(self,func,paras_init,args = args,bounds = bounds,**kwargs)
        self._method = kwargs.get("method","simplex")
        if self._method == "simplex":
            self._method = "nelder_mead"
        self._interface = _mloops_interface(self._func,args = args)
        
        i = len(self._bounds)
        min_bound = np.empty([i])
        max_bound = np.empty_like(min_bound)
        for j in range(i):
            min_bound[j] = (self._bounds[j])[0]
            max_bound[j] = (self._bounds[j])[1]
            
        self._controller = mlc.create_controller(self._interface, 
                                       controller_type = self._method,
                                       max_num_runs = self._max_run,
                                       max_num_runs_without_better_params = int(self._max_run/2),
                                       target_cost = self._target,
                                       num_params = len(self._paras_init), 
                                       first_params = self._paras_init,
                                       min_boundary = min_bound,
                                       max_boundary = max_bound, 
                                       trust_region = 0.2,
                                       **extra_dict
                                       )
        
    def optimization(self):
        self._controller.optimize()
        x_optimize = self._controller.best_params
        
        return x_optimize
        
    def visualization(self):
        self._visualization(self._flist,self._x_vec,self._method)
    
### operation
def main():
    def func(x,a,b,c,d):
        vec = np.array([a,b,c,d])
        f = np.sum((x - vec)**2,axis = None) + 5*np.sum(np.cos(x-a) + np.cos(x-b) + np.sin(x-c) + np.sin(x-d)) + a*b*c*d
        uncer = 0.1
        bad = None
        return_dict = {'cost':f,'uncer':uncer,'bad':bad}
        return return_dict
    
    method = "gaussian_process"
    
    init = np.array([3,0,4,2])
    a = 6
    b = 8
    c = 1
    d = 2
    bounds = ((-10,10),(-10,10),(-10,10),(-10,10))
    opt = mloop_optimize(func,init,args = (a,b,c,d,),bounds = bounds,max_run = 300,delay = 0.03,method = method,val_only = True)
    x_end = opt.optimization()
    print(x_end)
    opt.visualization()

if __name__ == '__main__':
    main()
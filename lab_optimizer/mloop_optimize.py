import mloop.interfaces as mli
import mloop.controllers as mlc
# import mloop.visualizations as mlv
from optimize_base import *
import numpy as np

class _mloops_interface(mli.Interface):
    def __init__(self,func,args):
        mli.Interface.__init__(self)
        self._func = func
        self._args = args
    
    #the method that runs the experiment given a set of parameters and returns a cost
    def get_next_cost_dict(self,params_dict):

        #The parameters come in a dictionary and are provided in a numpy array
        params = params_dict['params']
        cost_dict = self._func(params,*self._args)
        
        #The cost, uncertainty and bad boolean must all be returned as a dictionary
        return cost_dict
    
class mloop_optimize(base_optimizer):
    """reconstructed mloop algorithms <https://m-loop.readthedocs.io/en/stable/index.html> 
        mloop is a good integrated lab used optimization algorithm, ``methods`` including:
        - ``'gaussian_process', 'neural_net', 'differential_evolution',  'simplex', 'random'``

        Warning
        ---------
        in mloop_optimize, the func is required to return a valid uncer and bad and val_only is automatically set to False  
        
        Args
        ---------
        fun : callable
            The objective function to be minimized.

                ``fun(x, *args) -> dict : {'cost':float, 'uncer':float, 'bad':bool}``
                
            where ``cost`` is the value to minimize (nan will raise an error), ``uncer`` is uncertainty of cost,
            ``bad`` is the judge whether this value is bad (True means bad) for this cost
            

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

                Should be Sequence of ``(min, max)`` pairs for each element in `x`. None is used to specify no bound.
        
        kwArgs
        ---------
        ave_dict : dict
            - ave : Bool
                whethr to use average
            - ave_times : int
                average times
            - ave_wait
                wait times during each ave_run
            - ave_opc
                average operation code, defeault is "ave"
                - "ave" : following cost_dict
                - "std" : use for val_only func, it will cal uncer automatedly
                
            defeault is {False, X, X, X}
            if you set ave == True, then defeault is {True, 3, 0.01,"ave"}
            
        extra_dict : dict
            used for extra parameters for mloop controller
        
        method : string 
            method of scipy.optimize.minimize to be used, 
            should be one of: 'gaussian_process', 'neural_net', 'differential_evolution',  'simeplx' which is "nelder_mead", 'random'
        
        delay : float 
            delay of each iteration, default is 0.1s
        
        max_run : int 
            maxmun times of running optimization, default = 100
        
        msg : Bool
            whether to output massages in every iterarion, default is True
            
        log : Bool
            whether to generate a log file in labopt_logs
            
        logfile : str
            log file name , defeault is "optimization__ + <timestamp>__ + <method>__.txt"
            level lower than inherited logfile
            
        opt_inherit : class 
            inherit ``optimization results``, ``parameters`` and ``logs``
            defeault is None (not use inherit)
        
        Example
        ---------
        do not use opt_inherit
        >>> from lab_optimizer import mloop_optimize
        >>> opt1 = mloop_optimize(func,paras_init,bounds,args)
        >>> x_opt = opt.optimization()
        >>> opt.visualization()
        \\
        use opt_inherit (cascade multi optimizers)
        >>> from lab_optimizer import mloop_optimize
        >>> opt1 = mloop_optimize(func,paras_init,bounds,args,log = "inherit")
        >>> x_opt1 = opt.optimization()
        >>> # x_opt1 = opt.x_optimize
        >>> opt2 = mloop_optimize(func,x_opt1,bounds,args,opt_inherit = opt1) # paras_init will be automatically set to x_opt1 
        >>> opt2.optimization()
        >>> opt2.visualization()
    """
    @staticmethod
    def _doc() -> str:
        doc = "mloop_optimizer"
        return doc
    
    def __init__(self,func:callable,paras_init:np.ndarray,bounds:tuple,args:tuple = (),extra_dict:dict = {},opt_inherit = None,**kwargs):
        kwargs["val_only"] = False # let f return cost dict instead of a cost value
        kwargs["msg"] = None # use mloop msg
        base_optimizer.__init__(self,func,paras_init.copy(),args = args,bounds = bounds,**kwargs,_opt_type = self._doc(),extra_dict = extra_dict,opt_inherit = opt_inherit)
        self._method = kwargs.get("method","gaussian_process")
        self._method = "nelder_mead" if self._method == "simplex" else self._method
        self._interface = _mloops_interface(self._func,args = self._args)
        
        i = len(self._bounds)
        min_bound = np.empty([i])
        max_bound = np.empty_like(min_bound)
        for j in range(i):
            min_bound[j] = (self._bounds[j])[0]
            max_bound[j] = (self._bounds[j])[1]
            
        self._controller = mlc.create_controller(self._interface, 
                                       controller_type = self._method,
                                       max_num_runs = self._max_run,
                                       max_num_runs_without_better_params = self._max_run//2,
                                       cost_has_noise = True,
                                       target_cost = -1e10,
                                       num_params = len(self._paras_init), 
                                       first_params = self._paras_init,
                                       min_boundary = min_bound,
                                       max_boundary = max_bound, 
                                       trust_region = 0.4,
                                       **extra_dict
                                       )
        
    def optimization(self):
        self._controller.optimize()
        self.x_optimize = self._controller.best_params
        
        print("******************************************")
        print("best parameters find : ")
        print(self.x_optimize)
        print("cost : ")
        self._func(self.x_optimize,*self._args)
        print("******************************************")
        
        self._logging()
        return self.x_optimize
    
### operation
def _main():
    from opt_lib.test_functions import F5 as FF
    def f_dec(func):
        def wrap(x,*args,**kwargs):
            f=func(x,*args,**kwargs)
            return dict(cost = f)
        return wrap

    func = f_dec(FF)
    method = "simplex"

    init = np.array([1.2,-1.1,1.4])
    bounds = ((-2,2),(-2,2),(-2,2))
    extra_dict = {} #dict(pop = 6,local_polish = False)
    opt = mloop_optimize(func,init,args = (),bounds = bounds,max_run = 100,delay = 0.01,method = method,extra_dict=extra_dict, log = True)
    opt.optimization()
    opt.visualization("all")

if __name__ == '__main__':
    _main()
    
del _main
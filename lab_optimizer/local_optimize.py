from optimize_base import *
from scipy.optimize import minimize
import numpy as np
from opt_lib import __optlib_local__ as optlib_local

class local_optimize(optimize_base):
    """reconstructed scipy.optmize.minimize, which is a ``local optimization algorithm`` we recommend using 
    - ``"simplex" - Nelder_Mead``, ``"Powell"``, ``"CG"``, ``BFGS``, ``L-BFGS-B``,``"TNC", "COBYLA", "COBYQA", "SLSQP"``,
    they can be directly called, no foreced extra needs in extra_dict
    
        warning
        --------
        "Newton-CG" needs an extra argument Hessian in extra_dict \\
        "trust-krylov", "trust-exact", "trust-ncg", "dogleg" requires Jabobian in extra_dict
    
        Args
        --------
        fun : callable
            The objective function to be minimized.

                ``fun(x, *args) -> dict : {'cost':float, 'uncer':float, 'bad':bool}``
                
            where ``cost`` is the value to minimize, ``uncer`` is uncertainty,
            ``bad`` is the judge whether this value is bad (bad = True) for this cost
            
            f you set val_only = True, then you can set bad and uncer to anything because they will not be used and default is True

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
            Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell,
            trust-constr, COBYLA, and COBYQA methods. There are two ways to specify
            the bounds:

                1. Instance of `Bounds` class.
                2. Sequence of ``(min, max)`` pairs for each element in `x`. None is used to specify no bound.
        
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
            used for extra parameters for scipy.optimize.minimize family such as jac, hessel ... 
        
        method : str or callable, optional
            Type of solver.  Should be one of

                - 'simplex' which is Nelder-Mead :ref:`(see here) <optimize.minimize-neldermead>`
                - 'Powell'      :ref:`(see here) <optimize.minimize-powell>`
                - 'CG'          :ref:`(see here) <optimize.minimize-cg>`
                - 'BFGS'        :ref:`(see here) <optimize.minimize-bfgs>`
                - 'Newton-CG'   :ref:`(see here) <optimize.minimize-newtoncg>`
                - 'L-BFGS-B'    :ref:`(see here) <optimize.minimize-lbfgsb>`
                - 'TNC'         :ref:`(see here) <optimize.minimize-tnc>`
                - 'COBYLA'      :ref:`(see here) <optimize.minimize-cobyla>`
                - 'COBYQA'      :ref:`(see here) <optimize.minimize-cobyqa>`
                - 'SLSQP'       :ref:`(see here) <optimize.minimize-slsqp>`
                - 'trust-constr':ref:`(see here) <optimize.minimize-trustconstr>`
                - 'dogleg'      :ref:`(see here) <optimize.minimize-dogleg>`
                - 'trust-ncg'   :ref:`(see here) <optimize.minimize-trustncg>`
                - 'trust-exact' :ref:`(see here) <optimize.minimize-trustexact>`
                - 'trust-krylov' :ref:`(see here) <optimize.minimize-trustkrylov>`
                - custom - a callable object, see below for description.
        
            default is "Nelder-Mead"
        
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
        >>> from lab_optimizer import local_optimize
        >>> opt1 = local_optimize(func,paras_init,bounds,args)
        >>> x_opt = opt.optimization()
        >>> opt.visualization()
        \\
        use opt_inherit (cascade multi optimizers)
        >>> from lab_optimizer import local_optimize
        >>> opt1 = local_optimize(func,paras_init,bounds,args,log = "inherit")
        >>> x_opt1 = opt.optimization()
        >>> # x_opt1 = opt.x_optimize
        >>> opt2 = local_optimize(func,x_opt1,bounds,args,opt_inherit = opt1) # paras_init will be automatically set to x_opt1
        >>> opt2.optimization()
        >>> opt2.visualization()
     
    """
    @staticmethod
    def _doc() -> str:
        doc = "local_optimizer"
        return doc
    
    def __init__(self,func,paras_init:np.ndarray,bounds:tuple,args:tuple = (),extra_dict:dict = {},opt_inherit = None,**kwargs):
        kwargs["val_only"] = True # only need cost
        kwargs["opt_inherit"] = opt_inherit
        optimize_base.__init__(self,func,paras_init,args = args,bounds = bounds,**kwargs,_opt_type = self._doc())
        self._extra_dict = extra_dict
        self._method = kwargs.get("method","simplex")
        if self._method == "simplex":
            self._method = "Nelder-Mead"
    
    def optimization(self):
        _local_opt__ = ["Nelder-Mead","Powell","CG","BFGS","Newton-CG","L-BFGS-B","TNC","COBYLA","COBYQA"
                 ,"SLSQP","trust-constr","dogleg","trust-ncg","trust-exact","trust-krylov","custom"]
        if self._method in _local_opt__: 
            res = minimize(self._func,self._paras_init,args = self._args,method = self._method,bounds = self._bounds,**self._extra_dict,options = {"maxiter":self._max_run})
            self.x_optimize = res.x
        else: ## opt_extension
            from opt_lib import get_method
            alg = get_method(self._method)
            res = alg(self._func,self._paras_init,args = self._args,bounds = self._bounds,**self._extra_dict)
            res.run()
            self.x_optimize = res.x
        
        print("******************************************")
        print("best parameters find : ")
        print(self.x_optimize)
        print("cost : ")
        self._func(self.x_optimize,*self._args)
        print("******************************************")
        
        self._logging()
        return self.x_optimize

def _main():
    def func(x,a,b,c,d):
        vec = np.array([a,b,c,d])
        f = 2*np.sum((x - vec)**2,axis = None) + 10*np.sum(np.cos(x-a)*np.cos(x-b) + np.sin(x-c) + np.sin(x-d)) + a*b*c*d # + 5*np.random.randn()
        uncer = 0.1
        bad = False
        return_dict = {'cost':f,'uncer':uncer,'bad':bad}
        return return_dict

    method1 = "test"
    ave_dict = {"ave":True,"ave_time":3,"ave_wait":0.01}

    init = np.array([3,-8,4,2])
    a = 6
    b = 8
    c = 1
    d = 2
    bounds = ((-10,10),(-10,10),(-10,10),(-10,10))
    opt1 = local_optimize(func,init,args = (a,b,c,d,),bounds = bounds,max_run = 128,delay = 0.0001,method = method1,val_only = True,ave_dict = ave_dict, log = True ,msg = True)
    opt1.optimization()
    
    # opt1.visualization()
    # opt2 = local_optimize(func,init,args = (a,b,c,d,),bounds = bounds,max_run = 10,delay = 0.002,method = method2,val_only = True,ave_dict = ave_dict, log = True,msg = True,opt_inherit = opt1)
    # x_end = opt2.optimization()
    # print(x_end)
    # opt2.visualization()

if __name__ == "__main__":
    _main()
    
del _main
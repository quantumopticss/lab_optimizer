from optimize_base import *
import numpy as np
from opt_lib import __optlib_global__ as optlib_global

class global_optimize(base_optimizer):
    """reconstructed global optmization algorithms:
        ``"dual_annealing", "differential_evolution", "direct", "shgo", "particle_swarm", "genetic"``
        we recommend using ``"dual_annealing", "differential_evolution", "direct"`` which are more efficient
        
        where ``direct`` doesn't need initial_params but requires an extra argument called eps in extra_dict,
        which represent serching step, which can't be too low, defeault is 1e-1 
        
            - dual_annealing : <scipy.optimize.dual_annealing>

                - if you set ``no_local_search = True`` in ``extra_dict``, then ``dual_annealing`` will degenerate to ``simulated_annealing``, \
                  defeault is False and using eps (defeault 0.1) to local gradiant descent
                    
            - differential_evolution : <scipy.optimize.differential_evolution>
            - shgo : <scipy.optimize.shgo>
            - direct : <scipy.optimize.direct>
            - particle_swarm : <scikit.PSO>
            - genetic : <scikit.GA>
            - artificial_fish: <scikit.AF>

        warning 
        --------- 
        global optimization algorithms (except "direct") do not need too many rounds, usually x ~ 5, because in each round the function will be called many times. 
        scikit-optimize ``genetic``, ``particle_swarm`` may be less efficient and less robust than scipy.optimization, and ``artificial_fish`` is very expensive for analog cost_func
        
        Args
        ---------
        fun : callable
            The objective function to be minimized.

                ``fun(x, *args) -> dict : {'cost':float, 'uncer':float, 'bad':bool}``
                
            where ``cost`` is the value to minimize (nan will raise an error), ``uncer`` is uncertainty of cost,
            ``bad`` is the judge whether this value is bad (True means bad) for this cost
            
            f you set val_only = True, then ``bad`` and ``uncer`` will not be used and default is val_only = True

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
                
            defeault is {False, X, X}
            if you set ave == True, then defeault is {True, 3, 0.01,"ave"}       

        method : string
            which global algorithm to use, should be one of
            
            - ``"dual_annealing"``
            - ``"differential_evolution"``
            - ``"direct"
            - ``"shgo"``
            - ``"genetic"``
            - ``"particle_swarm"``
            - ``"artificial_fish"(not recommend)``
            
            defeault is ``"dual_annealing"`` 
        
        extra_dict : dict
            used for extra parameters for optimization algorithms
        
        delay : float 
            delay of each iteration, default is 0.1s
        
        max_run : int 
            maxmun times of running optimization, default = 10 
        
        msg : Bool
            whether to output massages in every iterarion, default is True
        
        log : Bool
            whether to generate a txt log file in labopt_logs

        logfile : str
            log file name , defeault is "optimization__ + <timestamp>__ + <method>__.txt"
            level lower than inherited logfile
            
        opt_inherit : class 
            inherit ``optimization results``, ``parameters`` and ``logs``
            defeault is None (not use inherit)
        
        Example
        ---------
        do not use opt_inherit
        >>> from lab_optimizer import global_optimize
        >>> opt1 = global_optimize(func,paras_init,bounds,args)
        >>> x_opt = opt.optimization()
        >>> opt.visualization()
        \\
        use opt_inherit (cascade multi optimizers)
        >>> from lab_optimizer import global_optimize
        >>> opt1 = global_optimize(func,paras_init,bounds,args,log = "inherit")
        >>> x_opt1 = opt.optimization()
        >>> # x_opt1 = opt.x_optimize
        >>> opt2 = global_optimize(func,x_opt1,bounds,args,opt_inherit = opt1) # paras_init will be automatically set to x_opt1 
        >>> opt2.optimization()
        >>> opt2.visualization()

    """
    @staticmethod
    def _doc() -> str:
        doc = "global_optimizer"
        return doc
    
    def __init__(self,func:callable,paras_init:np.ndarray,bounds:tuple,args:tuple = (),extra_dict:dict = {},opt_inherit = None,**kwargs):
        kwargs["val_only"] = True # only need cost
        self._method = kwargs.get("method","simulated_annealing")
        kwargs["max_run"] = np.min([50,kwargs.get("max_run",10)])
        base_optimizer.__init__(self,func,paras_init.copy(),args = args,bounds = bounds,**kwargs,_opt_type = self._doc(),extra_dict = extra_dict,opt_inherit = opt_inherit)
        self._extra_dict = extra_dict
    
    #### scipy algorithms
    def _optimization_scipy(self):
        match self._method:
            case "shgo":
                from scipy.optimize import shgo
                
                ## lab_opt defeault sets 
                self._extra_dict["minimizer_kwargs"] = self._extra_dict.get("minimizer_kwargs",{"method":"L-BFGS-B","options":{"eps":self._extra_dict.get("eps",0.1)}})
                if "eps" in self._extra_dict:
                    del self._extra_dict["eps"]
                if "no_local_search" in self._extra_dict:
                    del self._extra_dict["no_local_search"]
                ##
                self._res = shgo(self._func,self._bounds,args = self._args,iters = self._max_run,options = {"maxfev":self._max_run},**self._extra_dict)
            
            case "differential_evolution":
                from scipy.optimize import differential_evolution
                
                ## lab_opt defeault sets 
                if "eps" in self._extra_dict:
                    del self._extra_dict["eps"]
                if "no_local_search" in self._extra_dict:
                    del self._extra_dict["no_local_search"]
                maxiter = self._max_run//(15*len(self._paras_init)) + 1
                ##
                self._res = differential_evolution(self._func,self._bounds,args = self._args,maxiter = maxiter,x0 = self._paras_init,**self._extra_dict)
        
            case "direct":
                from scipy.optimize import direct
                
                ## lab_opt defeault sets 
                if "no_local_search" in self._extra_dict:
                    del self._extra_dict["no_local_search"]
                self._extra_dict["eps"] = self._extra_dict.get("eps",0.1)
                self._extra_dict["locally_biased"] = self._extra_dict.get("locally_biased",False)
                self._extra_dict["vol_tol"] = self._extra_dict.get("vol_tol",1e-12)
                self._extra_dict["len_tol"] = self._extra_dict.get("len_tol",1e-3)
                ##
                self._res = direct(self._func,self._bounds,args = self._args,maxiter = self._max_run,**self._extra_dict)
        
            case "dual_annealing":
                self._method = "dual_annealing"
                from scipy.optimize import dual_annealing
                
                ## lab_opt defeault sets 
                self._extra_dict["minimizer_kwargs"] = self._extra_dict.get("minimizer_kwargs",{"method":"L-BFGS-B","bounds":self._bounds,"options":{"eps":self._extra_dict.get("eps",0.1)}})
                if "eps" in self._extra_dict:
                    del self._extra_dict["eps"]
                self._extra_dict["no_local_search"] =self._extra_dict.get("no_local_search",False) # defeault is to have local search
                self._extra_dict["initial_temp"] = self._extra_dict.get("initial_temp",6e3)
                self._extra_dict["accept"] = self._extra_dict.get("accept",-6.)
                ##
                self._res = dual_annealing(self._func,self._bounds,args = self._args,x0 = self._paras_init,maxiter = self._max_run,**self._extra_dict)
    
    #### scikit-opt algoriths ####
    def _optimization_scikit(self): 
        ## scikit optimize does not support args, but we can decorate the function
        def _func_args(x):
            func_args = self._func(x,*self._args)
            return func_args
    
        self._func_args = _func_args
        
        n_dim = len(self._paras_init)
        lb = np.empty(n_dim)
        ub = np.empty_like(lb)
        for i in range(n_dim):
            lb[i] = (self._bounds[i])[0]
            ub[i] = (self._bounds[i])[1]
            
        match self._method:
            case "particle_swarm":
                from sko.PSO import PSO
                if "eps" in self._extra_dict:
                    del self._extra_dict["eps"]
                if "no_local_search" in self._extra_dict:
                    del self._extra_dict["no_local_search"]
                self._extra_dict["size_pop"] = self._extra_dict.get("size_pop",15)  
                
                self._opt = PSO(self._func_args,n_dim = n_dim,max_iter = self._max_run,lb = lb, ub = ub,**self._extra_dict)
                self._opt.X[0,:] = self._paras_init
                
            case "genetic":
                from sko.GA import GA
                
                self._extra_dict["precision"] = self._extra_dict.get("eps",0.05)
                if "eps" in self._extra_dict:
                    del self._extra_dict["eps"]
                if "no_local_search" in self._extra_dict:
                    del self._extra_dict["no_local_search"]
                self._extra_dict["size_pop"] = self._extra_dict.get("size_pop",12)
                self._extra_dict["prob_mut"] = self._extra_dict.get("prob_mut",0.003)
                    
                self._opt = GA(self._func_args,n_dim = n_dim,max_iter = self._max_run,lb = lb, ub = ub,**self._extra_dict)
                
            case "artificial_fish":
                from sko.AFSA import AFSA
                if "eps" in self._extra_dict:
                    del self._extra_dict["eps"]
                if "no_local_search" in self._extra_dict:
                    del self._extra_dict["no_local_search"]
                    
                self._extra_dict["size_pop"] = self._extra_dict.get("size_pop",5)  
                self._opt = AFSA(self._func_args,n_dim = n_dim,max_iter = self._max_run//100,max_try_num = np.max([10,self._max_run//10]),**self._extra_dict)
                self._opt.X[0,:] = self._paras_init
                self._opt.Y[0,:] = self._func_args(self._paras_init)
    
    def optimization(self):
        if self._method in ["dual_annealing", "differential_evolution", "direct", "shgo"]: 
            self._optimization_scipy()
            self.x_optimize = self._res.x
        elif self._method in ["particle_swarm","genetic","artificial_fish"]:
            self._optimization_scikit()
            self.x_optimize, _ = self._opt.run()
        else: ## opt_extension
            from opt_lib import get_method
            alg = get_method(self._method)
            res = alg(self._func,self._paras_init,args = self._args,bounds = self._bounds,max_run = self._max_run,**self._extra_dict)
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
    from opt_lib.test_functions import F5 as FF
    def f_dec(func):
        def wrap(x,*args,**kwargs):
            f=func(x,*args,**kwargs)
            return dict(cost = f)
        return wrap

    func = f_dec(FF)
    method = "differential_evolution"

    init = np.array([30,-70,40])
    bounds = ((-200,200),(-200,200),(-200,200))
    # extra_dict = dict(pop = 10,local_polish = False)
    extra_dict = {}
    opt = global_optimize(func,init,args = (),bounds = bounds,max_run = 1,delay = 0.01,method = method,extra_dict=extra_dict, log = True)
    opt.optimization()
    opt.visualization("all")
    # from local_optimize import local_optimize
    # opt2 = local_optimize(func,init,args = (),bounds = bounds,max_run = 10,delay = 0.002,method = "L-BFGS-B",val_only = True, log = True,msg = True,opt_inherit = opt)
    # x_end = opt2.optimization()
    # print(x_end)
    # opt2.visualization()

if __name__ == "__main__":
    _main()
    
del _main

"""
direct algorithm is a deterministic global optimization algorithm particularly effective 
for problems where the objective function is expensive to evaluate or where derivative information is unavailable
a trade off bewteen global calculation and local refinement

shgo is also a determinestic algorithm, mesh the contrained region, calculate and 
get the next search region, a bit like simplex

differential_evolution, an advanced version of genetic

dual_annealing : an advanced version of simulated_annealing, random jump with finite possibility to accept
and accompanied local_optimize
"""

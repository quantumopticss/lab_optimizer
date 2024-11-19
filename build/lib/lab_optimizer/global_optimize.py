from .optimize_base import optimize_base
import numpy as np

class global_optimize(optimize_base):
    """reconstructed global optmization algorithms:
        ``"dual_annealing", "differential_evolution", "direct", "shgo", "particle_swarm", "genetic"``
        we recommend using ``"dual_annealing", "differential_evolution", "direct"`` which are more efficient
        
        where ``direct`` doesn't need initial_params but requires an extra argument called eps in extra_dict,
        which represent serching step, which can't be too low, defeault is 1e-1 
        
            - dual_annealing : <scipy.optimize.dual_annealing>
            - differential_evolution : <scipy.optimize.differential_evolution>
            - shgo : <scipy.optimize.shgo>
            - direct : <scipy.optimize.direct>
            - particle_swarm : <scikit.PSO>
            - genetic : <scikit.GA>
            - artificial_fish: <scikit.AF>

        ``warning`` : 
        global optimization algorithms (except "direct") do not need too many rounds, usually x ~ 5, because in each round the function will be called many times. 
        scikit-optimize "genetic", "particle_swarm" may be less efficient and less robust than scipy.optimization
        
        Args
        ---------
        fun : callable
            The objective function to be minimized.

                ``fun(x, *args) -> dict : {'cost':float, 'uncer':float, 'bad':bool}``
                
            where ``cost`` is the value to minimize, ``uncer`` is uncertainty,
            ``bad`` is the judge whether this value is bad (bad = True) for this cost
            
            if you set val_only = True, then you can set bad and uncer to anything because they will not be used and default is True

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
        method : string
            which global algorithm to use, should be one of
            ``"dual_annealing","differential_evolution","direct","shgo","genetic","particle_swarm","artificial_fish"``, \
            defeault is ``"dual_annealing"``
        
        extra_dict : dict
            used for extra parameters for scipy.optimize.dual_annealing
        
        bounds : sequence or `Bounds`, optional
            Bounds on variables
            
                should be Sequence of ``(min, max)`` pairs for each element in `x`. None is used to specify no bound.
        
        delay : float 
            delay of each iteration, default is 0.1s
        
        max_run : int 
            maxmun times of running optimization, default = 10 
        
        msg : Bool
            whether to output massages in every iterarion, default is True
        
        log : Bool
            whether to generate a log file in labopt_logs
            
    """
    def __init__(self,func,paras_init,args = (),extra_dict = {},bounds = None,**kwargs):
        kwargs["val_only"] = True # only need cost
        self._method = kwargs.get("method","simulated_annealing")
        if "max_rum" not in kwargs:
            kwargs["max_run"] = 10
        optimize_base.__init__(self,func,paras_init,args = args,bounds = bounds,**kwargs)
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
        
            case _:
                self._method = "dual_annealing"
                from scipy.optimize import dual_annealing
                
                ## lab_opt defeault sets 
                self._extra_dict["minimizer_kwargs"] = self._extra_dict.get("minimizer_kwargs",{"method":"L-BFGS-B","bounds":self._bounds,"options":{"eps":self._extra_dict.get("eps",0.1)}})
                if "eps" in self._extra_dict:
                    del self._extra_dict["eps"]
                self._extra_dict["no_local_search"] =self._extra_dict.get("no_local_search",False) # defeault is no local search
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
                self._opt = PSO(self._func_args,n_dim = n_dim,max_iter = self._max_run,lb = lb, ub = ub,**self._extra_dict)
                self._opt.X = self._paras_init
                
            case "genetic":
                from sko.GA import GA
                eps = self._extra_dict.get("eps",0.05)
                if "eps" in self._extra_dict:
                    del self._extra_dict["eps"]
                self._opt = GA(self._func_args,n_dim = n_dim,max_iter = self._max_run,lb = lb, ub = ub,precision = eps,**self._extra_dict)
                self._opt.X = self._paras_init
                
            # case "artificial_fish":
            #     from sko.AFSA import AFSA
            #     if "eps" in self._extra_dict:
            #         del self._extra_dict["eps"]
            #     self._opt = AFSA(self._func_args,n_dim = n_dim,max_iter = self._max_run,max_try_num = np.max([50,self._max_run//4]),**self._extra_dict)
            #     self._opt.X[0,:] = self._paras_init
                
    def optimization(self):
        if self._method in ["particle_swarm","genetic","artificial_fish"]:
            self._optimization_scikit()
            x_optimize, _ = self._opt.run()
            print("best parameters find: ")
            print(self._func_args(x_optimize))
        else:
            self._optimization_scipy()
            x_optimize = self._res.x
            print("best parameters find: ")
            print(self._func(x_optimize,*self._args))
        
        return x_optimize
    
    def visualization(self):
        self._visualization(self._flist,self._x_vec,self._method)

def main():
    def func(x,a,b,c,d):
        vec = np.array([a,b,c,d])
        f = 0.01*np.sum((x - vec)**2*np.sum((x+0.3*vec)**2),axis = None) + 10*np.sum(np.cos(x-a) + np.cos(x-b) + np.sin(x-c) + np.sin(x-d)) + a*b*c*d
        uncer = 0.1
        bad = None
        return_dict = {'cost':f,'uncer':uncer,'bad':bad}
        return return_dict
    
    method = "dual_annealing"
    
    init = np.array([3,0,4,2])
    a = 6
    b = 8
    c = -5
    d = 2
    bounds = ((-10,10),(-10,10),(-10,10),(-10,10))
    extra_dict = {"no_local_search":True,"eps":np.array([0.2,0.2,0.2,0.2])}
    opt = global_optimize(func,init,args = (a,b,c,d,),bounds = bounds,max_run = 2,delay = 0.03,method = method,extra_dict=extra_dict,val_only = True, log = True)
    x_end = opt.optimization()
    print(x_end)
    opt.visualization()
     
if __name__ == "__main__":
    main()
        
            
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

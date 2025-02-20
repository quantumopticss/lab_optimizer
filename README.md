lab_optimizer, with latest version 1.3.1
---------
optimization algorithms packages

- Provides : 
  1. <span style="color:red">global_optimizer</span>, include algorithms for finding <span style="color:red">__global minimun__</span> 
  2. <span style="color:green">local_optimizer</span>, include algorithms for finding <span style="color:green">__local minimun__</span>
  3. mloop_optimizer, a general and integral API of its functions
  4. torch_optimizer __(only for torch functions)__, a general and integral API of its functions
  5. powerful visualization tools
  6. physics constants and units conversion 
  7. lab_optimizer examples
  8. **flexible custom opt algorithm API and visualization API**

- to download this package, using
```shell
pip install lab_optimizer
```

Quick Start
---------
- basic set : 
```python
## general form
opt = XXX_optimize(func,paras_init,bounds,args)
x_opt = opt.optimization() # x_opt is the optimization result
opt.visualization()
"""
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
    trust-constr, COBYLA, and COBYQA methods. To specify
    the bounds:

        Sequence of ``(min, max)`` pairs for each element in `x`. None is used to specify no bound.

kwArgs
---------
ave_dict : dict
    - ave : Bool
        whethr to use average
    - ave_times : int
        average times
    - ave_wait
        wait times during each ave_run
    - ave_opt
        average operation code, defeault is "ave"
        - "ave" : following cost_dict
        - "std" : use for val_only func, it will cal uncer automatedly
        
    defeault is {False, X, X, X}
    if you set ave == True, then defeault is {True, 3, 0.01,"ave"}

extra_dict : dict
    used for extra parameters for scipy.optimize.minimize family such as jac, hessel ... 

method : str 
    optimization algorithm to use

delay : float 
    delay of each iteration, default is 0.1s

max_run : int 
    maxmun times of running optimization

msg : Bool
    whether to output massages in every iterarion, default is True
    
log : Bool
    whether to generate a log file in folder labopt_logs
    
logfile : str
    log file name , defeault is "optimization__ + <timestamp>__ + <method>__.txt"
    level lower than inherited logfile
    
opt_inherit : opt_class 
    inherit ``optimization results``, ``parameters`` and ``logs``
    defeault is None (not use inherit)
"""

## get example code
from lab_optimizer import examples
examples(opcode = "direct_opt") # direct_opt, inherit_opt, log
```

- using examples : 

    - do not use opt_inherit
    ```python
    from lab_optimizer import global_optimize
    opt = global_optimize(func,paras_init,bounds,args)
    x_opt = opt.optimization()
    opt.visualization()
    ```

    - use opt_inherit (cascade multi optimizers)
    ```python
    from lab_optimizer import global_optimize
    opt1 = global_optimize(func,paras_init,bounds,args,log = "inherit")
    x_opt1 = opt1.optimization()
    # x_opt1 = opt.x_optimize ## you can also use this one
    opt2 = global_optimize(func,x_opt1,bounds,args,opt_inherit = opt1) # paras_init will be automatically set to x_opt1 
    opt2.optimization()
    opt2.visualization()
    ```

    - Generic functional interface
    
    to use another optimization algorithm, you only need to change the opt_class and just a little about its args
    ```python
    from lab_optimizer import global_optimize, local_optimize
    opt1 = global_optimize(func,paras_init,bounds,args,log = "inherit")
    x_opt1 = opt1.optimization()
    opt2 = local_optimize(func,x_opt1,bounds,args,opt_inherit = opt1) # just change opt_class from global_opt to local_opt
    opt2.optimization()
    opt2.visualization()
    ```

- units module 
```python
from lab_optimizer import units
"""
then you can easily use physics constants like :

- planck constant : units.h_const
- velocity of light in vacuum : units.c0_const

and do units conversion :  

- freq = 100*units.THz = 1e5*units.GHz = 1e8*units.MHz = ... # freq = 1e14 Hz
- m = 100*units.kg = 1e5*units.g # m = 100[kg]

the units module uses SI units (kg,m,s)
"""
```

Documentation
---------
- local_optimize : 
  
  local_optimize aims at finding local minimum of a function, the __local_optimize__ submodule is constructed based on [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html), including all of its supported algorithms : 

  - Nelder-Mead (defeault)
  - L-BFGS-B
  - Powell
  - ...

- global_optimize :
  
  global_optimize aims at finding global minimum of a function, the __global_optimize__ submodule is constructed based on [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) and [scikit-opt](https://scikit-opt.github.io/scikit-opt/#/en/README) including some powerful algorithms 
  
  - based on scipy.optimize 

    - [dual_annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html#scipy.optimize.dual_annealing) (defeault)
    - [differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html#scipy.optimize.differential_evolution)
    - [direct](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.direct.html#scipy.optimize.direct)
    - [shgo](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html#scipy.optimize.shgo)

  - [based on scikit-opt](https://scikit-opt.github.io/scikit-opt/#/en/README)
    
    - genetic
    - particle_swarm
    - artificial_fish

- mloop_optimize :

  mloop_optimize inherits all functions of [M-LOOP](https://m-loop.readthedocs.io/en/stable/index.html) : 

  - gaussian_process (defeault)
  - neural_net
  - differential_evolution
  - Nelder-Mead
  - Random

- torch_optimize : 

  torch_optimize aims at optimizing __explicit function__ (which can be expressed explicitly in your code instead of experiment results), using [torch based gradient optimization algorithms family](https://pytorch.org/docs/stable/optim.html) 

  - 'ASGD' (defeault)
  - 'SGD'
  - 'RMSprop'
  - 'LBFGS' (require extra parameters in extra_dict)
  - 'Rprop'
  - 'Adadelta'
  - 'Adagrade' (cpu_only)
  - 'Adam'
  - 'NAdam'
  - 'RAdam'
  - 'AdamW'
  - 'Adamax'

- other build_in functions : 

  - local_time : get local time since the epoch, return (time.time() + time_zone*3600.0)
  - read_log : get flist and x_vec, which records cost and parameters during opt process
  - log_visual : view optimization results from log 
  - opt_random_seed : set random seeds for numpy and torch module to ensure the repeatability of experiments

ReleaseNotes
---------
- 1.1.x 

  add advanced visualization tools :

  - [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
  - [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
  - [parallel coordinates](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.plotting.parallel_coordinates.html)
  - [scatter matrix](https://seaborn.pydata.org/examples/scatterplot_matrix.html)

- 1.2.x

  add functions to handel optimizations Exceptions
  
  add optimizer extensions, provide a general interface of custom defined opt algorithms
  
  add optimizer test function library:
  
  add opt algorithms library:
    - Improved Slime Mould Algorithm (ISMA) @ global , [an introduction of Slime Mould Algorithm](https://en.wikiversity.org/wiki/Slime_Mould_Algorithm)

- 1.3.x(building)

  parallelizing in plotting figures, can save much time for large data visualization - finished

  add agent model for expensive func opt jobs - todo
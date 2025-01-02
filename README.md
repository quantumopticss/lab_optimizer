# lab_optimizer
lab_optimization algorithms packages
Provides : 
  1. global_optimizer
  2. local_optimizer
  3. mloop_optimizer
  4. torch_optimizer (only for torch functions)
  5. physics units and constants 
  #. lab_optimizer examples

to download this package, using

>>> pip install lab_optimizer

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


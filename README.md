lab_optimizer
---------
optimization algorithms packages

- Provides : 
  1. <span style="color:red">global_optimizer</span>, include algorithms for finding <span style="color:red">__global minimun__</span> 
  2. <span style="color:green">local_optimizer</span>, include algorithms for finding <span style="color:green">__local minimun__</span>
  3. mloop_optimizer, a general and integral API of its functions
  4. torch_optimizer (only for torch functions), a general and integral API of its functions
  5. physics constants and units conversion 
  6. lab_optimizer examples

- to download this package, using
```shell
pip install lab_optimizer
```

Examples
---------
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


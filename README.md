# lab_optimizer
lab_optimization algorithms packages

to download this package, using

>>> pip install lab_optimizer

to quickly use lab_optimizer : 

import lab_optimizer import XXX_optimize # XXX is sub opt package, including : global, local, mloop, torch \\
opt = XXX_optimize(func,paras_init,args) \\
opt.optimize() # run optimization algorithm \\
opt.visualization # visualize opt results \\


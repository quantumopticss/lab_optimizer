import os,sys
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))

import numpy as np
import lab_optimization as lab_opt

def main_numpy():
    def func(x,a,b,c,d):
        vec = np.array([a,b,c,d])
        f = np.sum((x - vec)**2,axis = None) + 5*np.sum(np.cos(x-a) + np.cos(x-b) + np.sin(x-c) + np.sin(x-d)) + a*b*c*d
        uncer = 0.1
        bad = None
        return_dict = {'cost':f,'uncer':uncer,'bad':bad}
        return return_dict
    
    method = "dual_annealing"
    
    init = np.array([9.6825, -1.2480, -6.6065, -3.8342])
    a = 6
    b = 4
    c = 1
    d = 2
    bounds = ((-10,10),(-10,10),(-10,10),(-10,10))
    extra_dict = {} # {"eps":1e-1}
    opt = lab_opt.global_optimize(func,init,args = (a,b,c,d,),bounds = bounds,max_run = 1,delay = 0.03,method = method,extra_dict=extra_dict,val_only = True,msg = False,log = True)
    x_end = opt.optimization()
    print(x_end)
    opt.visualization()
    
def main_torch():
    import torch as th
    def func(x,a,b,c,d):
        vec = th.tensor([a,b,c,d])
        f = th.sum((x - vec)**2,dim = None) + 5*th.sum(th.cos(x-a) + th.cos(x-b) + th.sin(x-c) + th.sin(x-d)) + a*b*c*d
        uncer = 0.1
        bad = None
        return_dict = {'cost':f,'uncer':uncer,'bad':bad}
        return return_dict
    
    method = "AdamW"
    
    init = th.tensor([3.,0.,4.,2.])
    a = 6.
    b = 4.
    c = 1.
    d = 2.
    bounds = ((-10,10),(-10,10),(-10,10),(-10,10))
    extra_dict = {}
    opt = lab_opt.torch_optimize(func,init,args = (a,b,c,d,),bounds = bounds,max_run = 100,delay = 0.03,method = method,extra_dict=extra_dict)
    x_end = opt.optimization()
    print(x_end)
    opt.visualization()
     
if __name__ == "__main__":
    # main_numpy()
    main_torch()
    
    # # visual logs
    # path = "labopt_logs/lab_opt_2024_11/optimization__2024-11-17-13-14__dual_annealing__.txt"
    # lab_opt.log_visiual(path)
        
    # # """
    

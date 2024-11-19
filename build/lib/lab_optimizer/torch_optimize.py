## gradient descent
import torch as th
from torch import nn
import numpy as np
from optimize_base import optimize_base
from torch.optim.lr_scheduler import ExponentialLR

class _torch_interface(nn.Module):
    def __init__(self,func,paras_init,args):
        super().__init__()
        self._func1 = func
        self._args1 = args
        self._th_params = nn.Parameter(paras_init,requires_grad = True)
        
    def forward(self):
        cost = self._func1(self._th_params,*self._args1)
        return cost
    
class torch_optimize(optimize_base):
    """reconstructed pytorch ``gradient descent algorithm family``
    
            - 'ASGD' 
            - 'SGD'
            - 'RMSprop'
            - 'Adam'
            - 'AdamW'
            - 'Adamax'
            - 'Adagrade'
    
    require ``torch based func`` and must be ``based on explicite cost function``
        
    needs parameters about ``"lr", "lr_ctl"``, which represent learning rate and learning rate control
        
        Attention
        ---------
        all parameters should be float or complex, number must be represented as 1., 2.
        
        loss : ``tensor``
            Must be explicitly expressed by parameters, instead of some measurements by physics system.
            
                loss = th.max(th.abs(params)**2,dim = None) 
        
        Args
        ---------
        fun : ``torch callable``
            The objective function to be minimized.

                ``fun(x, *args) -> dict : {'cost':th.float, 'uncer':th.float, 'bad':bool}``
                
            where ``cost`` is the value to minimize, ``uncer`` is uncertainty,
            ``bad`` is the judge whether this value is bad (bad = True) for this cost
            
            if you set val_only = True, then you can set bad and uncer to anything because they will not be used and default is True

            ``x`` is a ``1-D tensor`` with shape (n,) and ``args``
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
            used for extra parameters for torch optimization algorithms
        
        lr : float
            learning rate, defeault is 0.05
            
        lr_clt : float
            learning rate control, regularly decrease learning rate
            defeault is 0.95
            
        method : string 
            method of scipy.optimize.minimize to be used, 
            should be one of: `` 'ASGD', 'SGD', 'RMSprop', 'Adam', 'AdamW', 'Adamax', 'Adagrade', ``
        
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
        kwargs["val_only"] = True # only need cose
        kwargs["torch"] = True # activate pytorch
        optimize_base.__init__(self,func,paras_init,args = args,bounds = bounds,**kwargs)
        self._method = kwargs.get("method","ASGD")
        self._model = _torch_interface(self._func,paras_init,args = args)
        
        match self._method:
            case "Adagrade":
                self._optimizer = th.optim.Adagrad(params = self._model.parameters(),lr = kwargs.get("lr",0.05),**extra_dict)
                
            case "Adamax":
                self._optimizer = th.optim.Adamax(params = self._model.parameters(),lr = kwargs.get("lr",0.05),**extra_dict)
            
            case "Adam":
                self._optimizer = th.optim.Adam(params = self._model.parameters(),lr = kwargs.get("lr",0.05),**extra_dict)
                
            case 'AdamW':
                self._optimizer = th.optim.AdamW(params = self._model.parameters(),lr = kwargs.get("lr",0.05),**extra_dict)
                
            case "RMSprop":
                self._optimizer = th.optim.RMSprop(params = self._model.parameters(),lr = kwargs.get("lr",0.05),**extra_dict)
                
            case "SGD":
                self._optimizer = th.optim.SGD(params = self._model.parameters(),lr = kwargs.get("lr",0.05),**extra_dict)
                                           
            case _:    
                self._method = "ASGD"
                self._optimizer = th.optim.ASGD(params = self._model.parameters(),lr = kwargs.get("lr",0.05),**extra_dict)
    
        self._scheduler = ExponentialLR(self._optimizer, gamma=kwargs.get("lr_ctl",0.95))
    
    def optimization(self):
        for n in range(self._max_run):
            self._model.train()
            
            _loss = self._model.forward()
            self._optimizer.zero_grad()
            _loss.backward()
            self._optimizer.step()
            
            self._model.eval()
            with th.inference_mode():
                if (n+1) % th.min(th.tensor([int(self._max_run/10), 500],dtype = th.int)) == 0:
                    self._scheduler.step()
                    
        x_optimize = self._model.state_dict()['_th_params']
        
        print("best parameters find: ")
        print(self._func(x_optimize,*self._args))
        
        return x_optimize
    
    def visualization(self):
        self._visualization(self._flist,self._x_vec,self._method)

def main():
    def func(x,a,b,c,d):
        vec = th.tensor([a,b,c,d])
        f = th.sum((x - vec)**2,dim = None) + 5*th.sum(th.cos(x-a) + th.cos(x-b) + th.sin(x-c) + th.sin(x-d)) + a*b*c*d
        uncer = 0.1
        bad = None
        return_dict = {'cost':f,'uncer':uncer,'bad':bad}
        return return_dict
    
    init = th.tensor([3.,0.,4.,2.])
    a = 6.;b=8.;c = 1.;d = 2.
    bounds = ((-10,10),(-10,10),(-10,10),(-10,10))
    # 'SGD', 'Adam','RMSprop','ASGD','AdamW', 'SparseAdam'
    opt = torch_optimize(func,init,args = (a,b,c,d),bounds = bounds,max_run = 100,delay = 0.07,method = "AdamW",lr = 0.1, lr_clt = 0.9,log = True)
    x_end =  opt.optimization()
    print(x_end)
    opt.visualization()
     
if __name__ == "__main__":
    main()
## gradient descent
import torch as th
from torch import nn
from optimize_base import *
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

            - 'ASGD' (defeault)
            - 'SGD'
            - 'RMSprop'
            - 'LBFGS' (require extra parameters in extra_dict)
            - 'Rprop'
            - 'Adadelta'
            - 'Adam'
            - 'NAdam'
            - 'RAdam'
            - 'AdamW'
            - 'Adamax'
            - 'Adagrade' (cpu_only)
    
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
            used for extra parameters for torch optimization algorithms except learning rate and learning rate control
        
        device : str
            working device of torch_optimize, defeault is try to use cuda
        
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

        logfile : str
            log file name , defeault is "optimization__ + <timestamp>__ + <method>__.txt"
            level lower than inherited logfile

        opt_inherit : class 
            inherit ``optimization results``, ``parameters`` and ``logs``
            defeault is None (not use inherit)
        
        Example
        ---------
        do not use opt_inherit
        >>> from lab_optimizer import torch_optimize
        >>> opt1 = torch_optimize(func,paras_init,bounds,args)
        >>> x_opt = opt.optimization()
        >>> opt.visualization()
        \\
        use opt_inherit (cascade multi optimizers)
        >>> from lab_optimizer import torch_optimize
        >>> opt1 = torch_optimize(func,paras_init,bounds,args,log = "inherit")
        >>> x_opt1 = opt.optimization()
        >>> # x_opt1 = opt.x_optimize
        >>> opt2 = torch_optimize(func,x_opt1,bounds,args,opt_inherit = opt1) # paras_init will be automatically set to x_opt1
        >>> opt2.optimization()
        >>> opt2.visualization()
        
    """
    @staticmethod
    def _doc() -> str:
        doc = "torch_optimizer"
        return doc
    
    def __init__(self,func,paras_init:th.Tensor,bounds:tuple = None,args:tuple = (),extra_dict:dict = {},opt_inherit = None,**kwargs):
        self._device = kwargs.get("device",("cuda" if th.cuda.is_available() else "cpu"))
        kwargs["val_only"] = True # only need cose
        kwargs["torch"] = True # activate pytorch
        kwargs["opt_inherit"] = opt_inherit
        kwargs["extra_dict"] = extra_dict
        optimize_base.__init__(self,func,paras_init,args = args,bounds = bounds,**kwargs,_opt_type = self._doc())
        self._method = kwargs.get("method","ASGD")
        self._model = _torch_interface(self._func,paras_init,args = args).to(self._device)
        
        match self._method:
            case "Adadelta":
                th_alg = th.optim.Adadelta
                
            case "RAdam":
                th_alg = th.optim.RAdam
                
            case "NAdam":
                th_alg = th.optim.NAdam
                
            case "LBFGS":
                th_alg = th.optim.LBFGS
                
            case "Rprop":
                th_alg = th.optim.Rprop
            
            case "Adagrade":
                th_alg = th.optim.Adagrad
                
            case "Adamax":
                th_alg = th.optim.Adamax
            
            case "Adam":
                th_alg = th.optim.Adam
                
            case 'AdamW':
                th_alg = th.optim.AdamW
                
            case "RMSprop":
                th_alg = th.optim.RMSprop
                
            case "SGD":
                th_alg = th.optim.SGD

            case _:    
                self._method = "ASGD"
                th_alg = th.optim.ASGD
                
        self._optimizer = th_alg(params = self._model.parameters(),lr = kwargs.get("lr",0.05),**extra_dict)
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
                    
        self.x_optimize = self._model.to("cpu").state_dict()['_th_params']
        
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
        vec = th.tensor([a,b,c,d])
        f = th.sum((x - vec)**2,dim = None) + 5*th.sum(th.cos(x-a) + th.cos(x-b) + th.sin(x-c) + th.sin(x-d)) + a*b*c*d
        uncer = 0.1
        bad = None
        return_dict = {'cost':f,'uncer':uncer,'bad':bad}
        return return_dict
    
    init = th.tensor([3.,0.,4.,-3.])
    a = 6.;b=8.;c = 1.;d = 2.
    bounds = ((-10,10),(-10,10),(-10,10),(-10,10))
    # 'SGD', 'Adam','RMSprop','ASGD','AdamW', 'SparseAdam'
    opt1 = torch_optimize(func,init,args = (a,b,c,d),bounds = bounds,max_run = 40,delay = 0.02,method = "RAdam",lr = 0.03, lr_clt = 0.9,log = "inherit")
    x_end =  opt1.optimization()
    opt1.visualization()
    # opt2 = torch_optimize(func,init,args = (a,b,c,d),bounds = bounds,max_run = 10,delay = 0.02,method = "SGD",lr = 0.05, lr_clt = 0.9,log = True,opt_inherit=opt1)
    # x_end = opt2.optimization()

    # opt2.visualization()
     
if __name__ == "__main__":
    _main()
    
del _main
class examples:    
    """ examples about using lab_optimizer,
    print example codes
    
    Args
    ---------
    opcode : string
        one of ``"direct_opt"`` or ``"inherit_opt"`` , ``"log"``
        
        - "direct_opt" : basic example about using lab_optimizer
        - "inherit_opt" : show how to using opt_inherit to do cascaded optimization
        - "log"`` : show how to review optimization log
        
    Example:
    >>> from lab_optimizer import opt_examples
    >>> opt_examples.examples("direct_opt")
    >>> " code example "        
    """
    def __init__(self,opcode:str = "direct_opt"):
        
        direct_opt = """
## examples for performing direct_opt
        
from lab_optimizer import global_optimize 
def func(x,a,b,c,d):
    vec = np.array([a,b,c,d])
    f = np.sum( (x-vec)**2 + 7*np.cos(2*np.pi*(x-vec))  )
    uncer = 0.1
    bad = None
    return_dict = {'cost':f,'uncer':uncer,'bad':bad}
    return return_dict

method = "dual_annealing"

init = np.array([3,0,4,2])
a = 6
b = 8
c = 5
d = 2
bounds = ((-10,10),(-10,10),(-10,10),(-10,10))
extra_dict = {"no_local_search":None,"eps":0.1}
opt = global_optimize(func,init,args = (a,b,c,d,),bounds = bounds,max_run = 1,delay = 0.03,method = method,extra_dict=extra_dict,val_only = True, log = True, logfile = "test_logfile")
x_end = opt.optimization()
opt.visualization()
        """
        
        inherit_opt = """
## example for performing inherit_opt
        
from lab_optimizer import local_optimize 
def func(x,a,b,c,d):
    vec = np.array([a,b,c,d])
    f = np.sum((x - vec)**2,axis = None) + 5*np.sum(np.cos(x-a) + np.cos(x-b) + np.sin(x-c) + np.sin(x-d)) + a*b*c*d + 5*np.random.randn()
    uncer = 0.1
    bad = False
    return_dict = {'cost':f,'uncer':uncer,'bad':bad}
    return return_dict

method1 = "simplex"
method2 = "CG"
ave_dict = {"ave":True,"ave_time":3,"ave_wait":0.01}

init = np.array([3,0,4,2])
a = 6
b = 8
c = 1
d = 2
bounds = ((-10,10),(-10,10),(-10,10),(-10,10))
opt1 = local_optimize(func,init,args = (a,b,c,d,),bounds = bounds,max_run = 100,delay = 0.02,method = method1,val_only = True,ave_dict = ave_dict, log = "inherit",msg = True)
opt1.optimization()
opt1.visualization()
opt2 = local_optimize(func,init,args = (a,b,c,d,),bounds = bounds,max_run = 10,delay = 0.02,method = method2,val_only = True,ave_dict = ave_dict, log = True,msg = True,opt_inherit = opt1)
x_end = opt2.optimization()
print(x_end)
opt2.visualization()
        """
        
        log = """
## example for visualizing opt_logs
        
from lab_optimizer import log_visual
path = "labopt_logs/lab_opt_2024_12_08/optimization__2024-12-08-15-59__ASGD__.txt"
log_visiual(path)
        """
        
        print("here is the using examples : ***")
        print("*********")
        match opcode:
            case "direct_opt":
                print(direct_opt)
            case "inherit_opt":
                print(inherit_opt)
            case "log":
                print(log)
        print("*********")
        

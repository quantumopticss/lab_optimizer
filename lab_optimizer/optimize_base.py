import numpy as np
import torch as th
import time 
import os
import matplotlib.pyplot as plt

"""
optimization base class

1.tool functions: including plot, timing and log processing
2.optimize base: including parameters set, function decorating, optimization progress visualizing 

"""

def multi_optimize(func,paras_init,args:tuple,optimizer_list:list,extra_dict_list:list,
                   method_list:list,max_run_list:list,bounds_list:list,**kwargs):
    """combine multi optimization algorithms
    
        Args
        --------
        fun : callable
            The objective function to be minimized.

                ``fun(x, *args) -> dict : {'cost':float, 'uncer':float, 'bad':bool}``
                
            where ``cost`` is the value to minimize, ``uncer`` is uncertainty,
            ``bad`` is the judge whether this value is bad (bad = True) for this cost

            ``x`` is a 1-D array with shape (n,) and ``args``
            is a tuple of the fixed parameters needed to completely
            specify the function.

        paras_init : ndarray, shape (n,)
            Initial guess. Array of real elements of size (n,),
            where ``n`` is the number of independent variables.

        args : tuple, optional
            Extra arguments passed to the objective function which will not
            change during optimization
            
        optimizer_list : list
            a ordered lists, in which are optimizers to be used 

        method_list : list
            a list, whose elements are ordered 
            optimization algorithm to use 

        extra_dict_list : list
            a list whose elements are ordered extra_dicts for optimizers,
            extra_dicts are used to transfer specific arguments for optimization algorithm
            
        bounds_list : list
            a list, whose elements are tuples,
            should be Sequence of ``(min, max)`` pairs for each element in `x`. None is used to specify no bound.
            
            >>> [((1,2),(1,2)),((1,2),(1,2)),((1,2),(1,2))]
            
            if len(bounds_list) != len(optimizers)
            
            will always use bounds_list[0]

        max_run_list : list
            a list, whose elements are ordered 
            maxmun times of running optimization, default = 10 

        kwArgs
        ---------
        delay : float 
            delay of each iteration, default is 0.1s
        
        msg : Bool
            whether to output massages in every iterarion, default is True
            
        log : Bool
            whether to generate a log file in labopt_logs
            
        logfile : str
            log file name , defeault is "optimization__ + <timestamp>__ + <method>__.txt"
            ``level lower than inherited logfile``
    """
    num_opt = len(optimizer_list)
    num_extra_dict = len(extra_dict_list)
    num_method = len(method_list)
    num_run = len(max_run_list)
    
    ## log name
    special_str = "["
    for opt_cls, str_method in zip(optimizer_list,method_list):
        special_str = special_str + opt_cls._doc() + "-" + str(str_method) + ";"
    special_str = special_str + "]"
    
    log = kwargs.get("log",None)
    kwargs["log"] = "alkaid"
    log_name = "cascated_opt__" + time.strftime("%Y-%m-%d-%H-%M",time.gmtime(local_time())) + "__" + special_str + "__" + ".txt"
    
    ## first run
    if num_opt != num_extra_dict or num_opt != num_method or num_opt != num_run:
        OptimizateException("all lists except bounds_list should have equal lens")
    
    optimizer = optimizer_list[0]
    try:
        opt_operator = optimizer(func,paras_init,args = args,bounds = bounds_list[0],**(extra_dict_list[0]),max_run = max_run_list[0],method = method_list[0],**kwargs,logfile = log_name)
    except:
        opt_operator = optimizer(func,paras_init,args = args,bounds = bounds_list[0],**(extra_dict_list[0]),max_run = max_run_list[0],method = method_list[0],**kwargs,logfile = log_name)
    
    paras_init = opt_operator.optimization()
    
    ## then
    for i in range(1,num_opt):
        ## add log in the last time
        if i == num_opt - 1:
            kwargs["log"] = log
        ## define opt class
        optimizer = optimizer_list[i]
        try:
            opt_operator = optimizer(func,paras_init,args = args,bounds = bounds_list[i],**(extra_dict_list[i]),
                                     method = method_list[i],max_run = max_run_list[i],
                                     **kwargs,opt_inherit = opt_operator)
        except:
            opt_operator = optimizer(func,paras_init,args = args,bounds = bounds_list[0],**(extra_dict_list[i]),
                                     method = method_list[i],max_run = max_run_list[i],
                                     **kwargs,opt_inherit = opt_operator)
        
        paras_init = opt_operator.optimization()

    ## visualization
    opt_plot(opt_operator._flist,opt_operator._x_vec,method_list)

def local_time(time_zone:int = 8) -> float:
    """get local time
    
        Args
        ---------
        time_zone : int
            local UTC time zone, defeault is 8

    """
    t = time.time() + time_zone*3600.0
    return t

def opt_plot(flist,x_vec,method,visual = "all"):
    if visual == "all":
        visual = ["traj","cost"]
    N,M = x_vec.shape
    ## cost vs rounds
    if "cost" in visual:
        plt.figure(1)
        timelist = np.arange(N)
        plt.plot(timelist,flist,label = "f value")
        plt.xlabel("rounds")
        plt.title("cost vs optimization rounds @ " + str(method))
        plt.legend()
    
    ## cost vs 
    if "traj" in visual:
        plt.figure(2)
        for i in range(M):
            plot_vec = x_vec[:,i]
            normal = np.max(np.abs(plot_vec),axis = None)
            plot_vec = plot_vec/normal
            plt.scatter(timelist,plot_vec,label = f"times vs paras-{i} with amp = {normal:.4f}")
        plt.legend()
        plt.xlabel("rounds")
        plt.title("normalized parameters  @ " + str(method))
        
        plt.figure(3)
        for i in range(M):
            plot_vec = x_vec[:,i]
            plt.scatter(timelist,x_vec[:,i],label = f"times vs paras-{i}")
        plt.legend()
        plt.xlabel("rounds")
        plt.title("raw parameters @ " + str(method))
    
    plt.show()
    
def log_visiual(path:str):
    """view optimization results from log  

        Args
        ---------
        path : string
            log path of optimization log 
        
    """
    def converter(s):
        s = s[1:-2].decode('utf-8')
        # Split the string into individual numbers and convert them to floats
        return np.array([float(x) for x in s.split(',')])

    print("logs : \n")
    head_numbers = 0
    with open(path, 'r', encoding='utf-8') as file:
        for currentline, line in enumerate(file, start=0):  # count from line 1
            f_msgs = line.strip()
            print(f_msgs)
            head_numbers += 1
            if f_msgs == "##":
                break
    
    data_list = np.loadtxt(path,skiprows = head_numbers,usecols=(2),converters = {2: converter},dtype = object)
    value_list = np.loadtxt(path,skiprows = head_numbers,usecols=(3))
    
    x_list = np.array([data_list[0]])
    for i in range(1,len(data_list)):
        x_list = np.vstack((x_list,data_list[i]))
    
    opt_plot(value_list,x_list,"from log : " + os.path.basename(path))

def ave_decorate(func,ave_times,ave_wait,ave_opt = "ave"):
    """average decorator:
    
    Args:
    ---------
        func : callable
            function to do average decoration, return a cost dict
            
        ave_times : int
            average times
            
        ave_wait : float   
            wait time during each average run
            
        ave_opt : str
            average operation code
            - "ave" to just follow cost_dict
            - "std" to support vals only result 
    """
    def ave_func(x,*args,**kwargs):
        cost = np.array([])
        if ave_opt == "ave": # follow cost_dict
            uncer = np.array([])
            bad = True
            for _ in range(int(ave_times)):
                f_dict = func(x,*args,**kwargs)
                cost = np.hstack((cost,f_dict["cost"]))
                uncer = np.hstack((cost,f_dict["uncer"]))
                try:
                    bad *= f_dict["bad"]
                except:
                    bad = False
                time.sleep(ave_wait)
            f_dict = {"cost":np.mean(cost),"uncer":np.sqrt(np.mean(uncer**2)),"bad" : bad}
        elif ave_opt == "std": # vals only 
            for _ in range(int(ave_times)):
                f_dict = func(x,*args,**kwargs)
                cost = np.hstack((cost,f_dict["cost"]))
                time.sleep(ave_wait)
            f_dict = {"cost":np.mean(cost),"uncer":np.std(cost),"bad" : False}
        
        return f_dict
    return ave_func

class OptimizateException(Exception):
    def __init__(self,err):
        Exception.__init__(self,"optimize error : " + err)
    
    ## user define
    @staticmethod    
    def user_define(func):
        def wrapper(self,*args,**kwargs):
            func(self,*args,**kwargs)
            raise OptimizateException(func.__name__ + " not defined!")
        return wrapper

class optimize_base(OptimizateException):
    """optimize_base class
    
    Args:
    ---------
        func : callable
            func to opt
            
        paras_init : np.ndarray || th.Tensor
            init parameters
            
        args : tuple
            extra args for func
            
        bounds : tuple
            bounds of opt algorithm
        
    Kwargs:
    ---------
        ave_dict : dict
            - ave : Bool
                whethr to use average
            - ave_times : int
                average times
            - ave_wait
                wait times during each ave_run
                
            defeault is {False, X, X}
            if you set ave == True, then defeault is {True, 3, 0.01}
    
        val_only : Bool
            whether to only use cost in cost_dict
            
        torch : Bool
            whether it is a torch opt class
        
        log : Bool
            whether to generate a log file
            
        log_file : str
            name of log name
            
        opt_ingerit : opt class
            ingerit some result from opt class result
            
    """
    def __init__(self,func,paras_init:np.ndarray,args:tuple = (),bounds:tuple = None,**kwargs):
        print("optimization start")
        self._time_start = local_time()
        self._args = args
        self._bounds = bounds
        
        ## not inherit args
        self._max_run = kwargs.get("max_run",100)
        self._val_only = kwargs.get('val_only',True)
        self._torch = kwargs.get("torch",False)
        self._log = kwargs.get("log", True)
        msg = kwargs.get("msg",True)
        delay = kwargs.get("delay",0.1)
        
        opt_inherit = kwargs.get("opt_inherit",None)
        ## inherit args
        if opt_inherit != None: # if we have inherit
            self._flist = opt_inherit._flist
            self._x_vec = opt_inherit._x_vec
            self._time_stamp = opt_inherit._time_stamp
            log_head_inhert = opt_inherit._log_head
            self._filename = opt_inherit._filename
            self._paras_init = opt_inherit.x_optimize
            self._run_count = opt_inherit._run_count
            self._ave_dict = opt_inherit._ave_dict
        else: # if no inherit
            self._paras_init = paras_init
            if self._torch == True:
                self._flist = th.tensor([(func(self._paras_init,*args).get("cost",0))])
                self._x_vec = self._paras_init.clone()
            else:   
                self._flist = np.array([(func(self._paras_init,*args).get("cost",0))])
                self._x_vec = np.array([self._paras_init])
            self._time_stamp = [time.strftime("%d:%H:%M:%S",time.gmtime(self._time_start))]
            log_head_inhert = ""
            self._filename = kwargs.get("logfile","optimization__" + time.strftime("%Y-%m-%d-%H-%M",time.gmtime(self._time_start)) + "__" + kwargs.get("method","None") + "__" + ".txt")
            self._ave_dict = kwargs.get("ave_dict",{"ave":False,"ave_times":1,"ave_wait":0.})
            self._run_count = 0
            
        ## using average
        if self._ave_dict.get("ave",False) == True:
            func = ave_decorate(func,self._ave_dict.get("ave_times",3),self._ave_dict.get("ave_wait",0.01))
        
        ## decorate func
        self._func = self._decorate(func,delay = delay,msg = msg)
            
        ## create log head
        if self._log == True or self._log == "inherit":
            self._log_head = log_head_inhert + (
                "start_time : " + time.strftime("%Y_%m_%d_%H:%M:%S",time.gmtime(local_time())) + " * " + "\n" +
                "method : " + kwargs.get("_opt_type","None") + " @ " + kwargs.get("method","None") + "\n" +
                "func : " + func.__repr__() + "\n" + 
                "args : " + self._args.__repr__() + "\n" +
                "paras_init : " + self._paras_init.__repr__() + "\n" +
                "kwargs : " + kwargs.__repr__() + "\n" + 
                "bounds : " + self._bounds.__repr__() + "\n" + 
                "max_run : "  f"{self._max_run}" + "\n"
                "form : " + "rounds, time, parameters, cost " + "\n\n" 
            )
        
    def _logging(self):
        self._time_end = local_time()
        delta_t = self._time_end - self._time_start
        f_delta_t = time.strftime("%H:%M:%S",time.gmtime(delta_t))
        print("\nthe optimization progress costs:")
        print(f"hh:mm:ss = {f_delta_t}\n")
        if self._log == True :
            ## folder
            os.makedirs("labopt_logs", exist_ok=True)
            sub_folder = os.path.join("labopt_logs","lab_opt_" + time.strftime("%Y_%m_%d",time.gmtime(self._time_start)) )
            os.makedirs(sub_folder, exist_ok=True)
            self._filename = os.path.join(sub_folder, self._filename)  # Store in a 'logs' directory

            ## head
            with open(self._filename, "w") as file:
                file.write("name : " + self._filename + "\n") 
                file.write(self._log_head)
                file.write("end_time : " + time.strftime("%Y_%m_%d_%H:%M:%S",time.gmtime(self._time_end)) + " * " +  "\n\n")
                file.write("##\n")
            ## data
            if type(self._x_vec) == th.Tensor:
                self._x_vec = self._x_vec.detach().numpy()
                self._flist = self._flist.deatch().numpy()
            with open(self._filename, "a") as file:
                for i in range(np.size(self._flist)):
                    file.write(f"{i}" + ", " +
                                self._time_stamp[i] 
                                + ", ")
                    file.write("[" + ",".join(map(str,self._x_vec[i])) + "]")
                    file.write(", " + f"{self._flist[i,0]}" + "\n")
    
    def _decorate(self,func,delay = 0.1,msg = True): # delay in s
        if self._torch == True:
            def func_decorate(x,*args,**kwargs):
                time.sleep(delay)
                f = func(x,*args,**kwargs)
                f_val = f.get("cost",0)
                print(f"INFO RUN: {self._run_count}")
                if msg == True:
                    print(f"INFO cost {f_val:.6f}")
                    print(f"INFO parameters {x}" + "\n")
                self._run_count += 1
                ## build flist including f values
                ## and x_vec in which x_vec[:,i] include the 
                ## changing traj of a parameter
                self._flist = th.vstack((self._flist,th.tensor([f_val])))
                self._x_vec = th.vstack((self._x_vec,x))
                self._time_stamp = self._time_stamp + [ time.strftime("%d:%H:%M:%S",time.gmtime(local_time())) ]
                if self._val_only == True:
                    return f_val
                else:
                    return f
        else:
            def func_decorate(x,*args,**kwargs):
                time.sleep(delay)
                f = func(x,*args,**kwargs)
                f_val = f.get("cost",0)
                print(f"INFO RUN: {self._run_count}")
                if msg == True:
                    print(f"INFO cost {f_val:.6f}")
                    print(f"INFO parameters {x}" + "\n")
                self._run_count += 1
                ## build flist including f values
                ## and x_vec in which x_vec[:,i] include the 
                ## changing traj of a parameter
                self._flist = np.vstack((self._flist,f_val))
                self._x_vec = np.vstack((self._x_vec,x))
                self._time_stamp = self._time_stamp + [ time.strftime("%d:%H:%M:%S",time.gmtime(local_time())) ]
                if self._val_only == True:
                    return f_val
                else:
                    return f
        return func_decorate

    @OptimizateException.user_define
    def optimization(self):
        """ you must define this method in XXX_optimize class
        """   
    
    def _visualization(self,flist,x_vec,method = "None",visual = "all"):        
        if type(x_vec) == th.Tensor:
            flist = flist.detach().numpy()
            x_vec = x_vec.detach().numpy()
        
        opt_plot(flist,x_vec,method,visual)
        
if __name__ == "__main__":
    
    # visual logs
    path = "labopt_logs/lab_opt_2024_12_08/optimization__2024-12-08-15-59__ASGD__.txt"
    log_visiual(path)
        
    # """
    
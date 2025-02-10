import numpy as np
import torch as th
import time 
import os
import matplotlib.pyplot as plt
# from typing import overload

"""
optimization base class

1.tool functions: including plot, timing and log processing
2.optimize base: including parameters set, function decorating, optimization progress visualizing 

"""

def local_time(time_zone:float = 8.0) -> float:
    """get local time, return (time.time() + time_zone*3600.0)
    
        Args
        ---------
        time_zone : float
            local UTC time zone, defeault is 8.

    """
    t = time.time() + time_zone*3600.0
    return t

def _opt_plot(flist,x_vec,method,visual = "all"):
    print("making visualizing figures")
    match visual:
        case "classic":
            visual = [visual]
        case "advanced":
            visual = [visual]
        case _:
            visual = ["classic","advanced"]
    N,M = x_vec.shape
    ## classic : traj & cost vs rounds
    if "classic" in visual:
        plt.figure(figsize=(12, 6))
        plt.subplot(1,2,1) # cost vs round
        timelist = np.arange(N)
        plt.plot(timelist,flist,label = "f value")
        plt.xlabel("rounds")
        plt.title("cost vs optimization rounds @ " + str(method))
        plt.legend()
        
        mean = np.mean(x_vec,axis = 0)        
        plt.subplot(1,2,2) # std-normal traj
        for i in range(M):
            plot_vec = x_vec[:,i]
            normal = np.std(plot_vec)
            plot_vec = (plot_vec - mean[i])/normal
            plt.scatter(timelist,plot_vec,label = f"times vs paras-{i} with : [amp-std = {normal:.4f} , mean = {mean[i]:.3f}]")
        plt.legend()
        plt.xlabel("rounds")
        plt.title("std normalized parameters  @ " + str(method))
    
    # advanced : higher dimensional visualizing
    if "advanced" in visual:
        import plotly.express as px
        import pandas as pd
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import MinMaxScaler
        import seaborn as sns 
        from pandas.plotting import parallel_coordinates
        
        # data 
        scalar = MinMaxScaler(feature_range=(0,255))
        df = pd.DataFrame(x_vec,columns=[f"x{i}" for i in range(x_vec.shape[1])])
        df['cost'] = flist
        
        # scalar x and f
        x_vec_scalar = scalar.fit_transform(x_vec)
        flist_scalar = scalar.fit_transform(flist)
        df_scalar = pd.DataFrame(x_vec_scalar,columns=[f"x{i}" for i in range(x_vec.shape[1])])
        df_scalar["cost"] = flist_scalar
        
        if x_vec.shape[1] <= 6:
            # Parallel Coordinates
            plt.figure(2,figsize=(6,6)) 
            df_scalar['cost_range'] = pd.qcut(df_scalar['cost'],q = 15,labels = False)

            parallel_coordinates(df_scalar, 'cost_range',colormap = "plasma")
            plt.title('Parallel Coordinates')
            plt.xlabel('ordered parameters [0-255] normalized')
            plt.ylabel('cost [0-255] normalized')  

            # scatter matrix
            sns.pairplot(df,vars = [f"x{i}" for i in range(x_vec.shape[1])],
                        hue = "cost", palette = 'viridis', diag_kind = 'hist',
                        plot_kws = {'alpha':0.8},height = 8.1/(x_vec.shape[1]))
            plt.title("scatter matrix")
        
        ## PCA
        pca = PCA(n_components=3)
        df['iter'] = range(1,len(df)+1)
        data_pca = pca.fit_transform(x_vec)
        
        df["PCA1"] = data_pca[:,0]
        df["PCA2"] = data_pca[:,1]
        df["PCA3"] = data_pca[:,2]

        # pca results
        variance_ratio = pca.explained_variance_ratio_
        comp = np.round(pca.components_,2)
        
        fig1 = px.scatter_3d(df,x = "PCA1",y = "PCA2",z = "PCA3",color = "cost",size="iter",
                             title = "high dimension visual @ PCA , focus on main dimension",
                             labels = {"cost":"cost"},hover_data = [f"x{i}" for i in range(x_vec.shape[1])])
        fig1.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey',colorscale='viridis')),opacity=0.8)
        
        fig1.update_layout(
            title="PCA 3D Scatter Plot",
            scene=dict(
                xaxis=dict(title="PCA1, comp = " + str(comp[0]) + f", contribute = {variance_ratio[0]*100:.2f}%"),
                yaxis=dict(title="PCA2, comp = " + str(comp[1]) + f", contribute = {variance_ratio[1]*100:.2f}%"),
                zaxis=dict(title="PCA3, comp = " + str(comp[2]) + f", contribute = {variance_ratio[2]*100:.2f}%")
                    )
            )
                
        ## t-SNE
        tsne = TSNE(n_components = 3,perplexity=np.min([30,2 + x_vec.shape[0]//11]),max_iter = 1000)
        data_tsne = tsne.fit_transform(x_vec)
        df["tsne1"] = data_tsne[:,0]
        df["tsne2"] = data_tsne[:,1]
        df["tsne3"] = data_tsne[:,2]
        fig2 = px.scatter_3d(df,x = "tsne1",y = "tsne2",z = "tsne3",color = "cost",size="iter",
                                title = "high dimension visual @ TSNE , focus on ** Clusters **",
                                labels = {"cost":"cost"},hover_data = [f"x{i}" for i in range(x_vec.shape[1])])
        fig2.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey',colorscale='viridis')),opacity=0.8)
        
        fig1.show()
        fig2.show()
        
    plt.show()

def log_visiual(path:str,visual:str = "all"):
    """view optimization results from log  

        Args
        ---------
        path : string
            log path of optimization log 
            
        visual : str
            to choose visualization figures, should be one of 
            
                - ``"classic"`` : to view just cost and std-normalized traj
                - ``"advanced"`` : provide multidimension visualization 
                - ``otherwise`` : all of them
            
            defeault is ``"all"``
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
    
    ## handling nan
    valid_flist = ~np.isnan(value_list)
    f_list = value_list[valid_flist]
    f_list = np.reshape(f_list,[-1,1])
    x_list = x_list[valid_flist.flatten(),:]
    
    _opt_plot(f_list,x_list,"from log : " + os.path.basename(path),visual)

def _ave_decorate(func,ave_times,ave_wait,ave_opc = "ave"):
    """average decorator:
    
    Args:
    ---------
        func : callable
            function to do average decoration, return a cost dict
            
        ave_times : int
            average times
            
        ave_wait : float   
            wait time during each average run
            
        ave_opc : str
            average operation code
            - "ave" to just follow cost_dict
            - "std" to support vals only result 
    """
    def ave_func(x,*args,**kwargs):
        cost = np.array([])
        if ave_opc == "ave": # follow cost_dict
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
        elif ave_opc == "std": # vals only 
            for _ in range(int(ave_times)):
                f_dict = func(x,*args,**kwargs)
                cost = np.hstack((cost,f_dict["cost"]))
                time.sleep(ave_wait)
            f_dict = {"cost":np.mean(cost),"uncer":np.std(cost),"bad" : False}
        
        return f_dict
    return ave_func

class optimize_Exception(Exception):
    def __init__(self,err_msg:str) -> Exception:
        """cls to handle exceptions
        
        Args
        ---------
        err_msg : str 
            error msgs
        """
        Exception.__init__(self,"optimize error : " + err_msg)
        
    @staticmethod
    def err_dict(err_key:str) -> str:
        """get err_msg corresponds to err_key

        Args
        ---------
            err_key : str
            
        """
        error_dict = dict(
            nan = "func return nan",
            not_def = "method not define",
            not_dict = "func should return cost dict : {cost,uncer,bad}"
        )
        return error_dict[err_key]

class optimize_base:
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
        - ave_opc
            average operation code, defeault is "ave"
            - "ave" : following cost_dict
            - "std" : use for val_only func, it will cal uncer automatedly
            
        defeault is {False, X, X, X}
        if you set ave == True, then defeault is {True, 3, 0.01}

    val_only : Bool
        whether to only use cost in cost_dict
        
    torch : Bool
        whether it is a torch opt class
    
    log : Bool
        whether to generate a log file
        
    log_file : str
        name of log name
        
    opt_inherit : class 
        inherit ``optimization results``, ``parameters`` and ``logs``
        defeault is None (not use inherit)
        
    """    
    def __init__(self,func:callable,paras_init:np.ndarray|th.Tensor,args:tuple = (),bounds:tuple = None,**kwargs):
        print("optimization start")
        
        ## not inherit args
        self._time_start = local_time()
        self._args = args
        self._bounds = bounds
        self._max_run = kwargs.get("max_run",100)
        self._val_only = kwargs.get('val_only',True)
        self._torch = kwargs.get("torch",False)
        self._log = kwargs.get("log", True)
        msg = kwargs.get("msg",True)
        delay = kwargs.get("delay",0.1)
        
        self.opt_inherit = kwargs.get("opt_inherit",None)
        ## inherit args
        if self.opt_inherit != None: # if we have inherit
            self._flist = self.opt_inherit._flist
            self._x_vec = self.opt_inherit._x_vec
            self._time_stamp = self.opt_inherit._time_stamp
            log_head_inhert = self.opt_inherit._log_head
            self._filename = self.opt_inherit._filename
            self._paras_init = self.opt_inherit.x_optimize
            self._run_count = self.opt_inherit._run_count
            self._ave_dict = self.opt_inherit._ave_dict
        else: # if no inherit
            self._paras_init = paras_init
            result = func(self._paras_init,*args)
            if type(result) != dict:
                self.error("not_dict")
            if self._torch == True:
                self._flist = th.tensor([result.get("cost",0)], device = self._device)
                self._x_vec = self._paras_init.clone()
            else:   
                self._flist = np.array([result.get("cost",0)])
                self._x_vec = np.array([self._paras_init])
            self._time_stamp = [time.strftime("%d:%H:%M:%S",time.gmtime(self._time_start))]
            log_head_inhert = ""
            self._filename = kwargs.get("logfile","optimization__" + time.strftime("%Y-%m-%d-%H-%M",time.gmtime(self._time_start)) + "__" + kwargs.get("method","None") + "__" + ".txt")
            self._ave_dict = kwargs.get("ave_dict",{"ave":False,"ave_times":1,"ave_wait":0.})
            self._run_count = 0
            
        ## create log head
        if self._log == True or self._log == "inherit":
            self._log_head = log_head_inhert + (
                "start_time : " + time.strftime("%Y_%m_%d_%H:%M:%S",time.gmtime(local_time())) + " * " + "\n" +
                "opt_alg : " + kwargs.get("_opt_type","None") + " @ " + kwargs.get("method","None") + "\n" +
                "func : " + func.__repr__() + "\n" + 
                "paras_init : " + self._paras_init.__repr__() + "\n" +
                "bounds : " + self._bounds.__repr__() + "\n" + 
                "args : " + self._args.__repr__() + "\n" +
                "kwargs : " + kwargs.__repr__() + "\n" 
            )

        # decorate func
        ## using average
        if self._ave_dict.get("ave",False) == True and self._torch == False:
            func = _ave_decorate(func,self._ave_dict.get("ave_times",3),self._ave_dict.get("ave_wait",0.01))
        
        ## normal decorate
        self._func = self._decorate(func,delay = delay,msg = msg)
    
    def _logging(self,err_msg:str = ""):
        """generating loggings
        """
        
        # if there are err_msg , we will add special head !!_ in log
        if err_msg != "":
            self._filename = "err_" + self._filename
            
        self._time_end = local_time()
        delta_t = self._time_end - self._time_start
        f_delta_t = time.strftime("%H:%M:%S",time.gmtime(delta_t))
        print("\nthe optimization progress costs:")
        print(f"hh:mm:ss = {f_delta_t}\n")
        
        if self._log == True or err_msg != "":
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
                if err_msg != "":
                    file.write("** " + "ERROR : " +  err_msg + " : ERROR" + " ** \n\n")
                file.write("form : " + "rounds, time, parameters, cost " + "\n\n")
                file.write("##\n")
            ## data
            if type(self._x_vec) == th.Tensor:
                self._x_vec = self._x_vec.to("cpu").detach().numpy()
                self._flist = self._flist.to("cpu").detach().numpy()
            with open(self._filename, "a") as file:
                for i in range(self._flist.size):
                    file.write(f"{i}" + ", " +
                                self._time_stamp[i] 
                                + ", ")
                    file.write("[" + ",".join(map(str,self._x_vec[i])) + "]")
                    file.write(", " + f"{self._flist[i,0]}" + "\n")
    
    def _decorate(self,func,delay = 0.1,msg = True):
        ## decorate optimization function:
        # msg -> message each call
        # delay -> delay each call
        # _torch -> whether torch func
        
        if self._torch == True:
            def func_decorate(x,*args,**kwargs):
                time.sleep(delay)
                f = func(x,*args,**kwargs)
                f_val = f.get("cost")
                print(f"INFO RUN: {self._run_count}")
                if msg == True:
                    print(f"INFO cost {f_val:.6f}")
                    print(f"INFO parameters {x}" + "\n")
                self._run_count += 1
                ## build flist including f values
                ## and x_vec in which x_vec[:,i] include the 
                ## changing traj of a parameter 
                self._flist = th.vstack((self._flist,th.tensor([f_val],device = self._device)))
                self._x_vec = th.vstack((self._x_vec,x))
                self._time_stamp = self._time_stamp + [ time.strftime("%d:%H:%M:%S",time.gmtime(local_time())) ]
                if th.isnan(f_val): # nan error
                    self.error("nan") 
                if self._val_only == True:
                    return f_val
                else:
                    return f
        else:
            def func_decorate(x,*args,**kwargs):
                time.sleep(delay)
                f = func(x,*args,**kwargs)
                f_val = f.get("cost")
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
                if np.isnan(f_val): # nan error
                    self.error("nan") 
                if self._val_only == True:
                    return f_val
                else:
                    return f
        return func_decorate
    
    ## developers are supposed to override this method for each sub_optimizer
    def optimization(self):...

    def visualization(self,visual:str = "all"):
        """to visualize optimization results
        
            Args
            ---------
            visual : str
                to choose visualization figures, should be one of 
                
                    - ``"classic"`` : to view just cost and std-normalized traj
                    - ``"advanced"`` : provide multidimension visualization 
                    - ``"all"`` : all of them
                
                defeault is ``"all"``
        """        
        if type(self._x_vec) == th.Tensor:
            self._flist = self._flist.to("cpu").detach().numpy()
            self._x_vec = self._x_vec.to("cpu").detach().numpy()
        
        _opt_plot(self._flist,self._x_vec,self._method,visual)
        
    def error(self,err:str) -> Exception:
        """ rasse an error
        
        Args
        ---------
        err : str
            error opc

        Raises
        ---------
        optimize_Exception : Exception

        """
        err_msg = optimize_Exception.err_dict(err)
        try:
            self._logging(err_msg)
        except:
            pass
        raise optimize_Exception(err_msg)

if __name__ == "__main__":
    path = "labopt_logs/lab_opt_2025_01_06/err_optimization__2025-01-06-17-37__simplex__.txt"
    log_visiual(path)
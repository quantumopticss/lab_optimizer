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

def local_time(time_zone:float = 8.0) -> float:
    """get local time since the epoch, return (time.time() + time_zone*3600.0)
    
        Args
        ---------
        time_zone : float
            local UTC time zone, defeault is 8.

    """
    t = time.time() + time_zone*3600.0
    return t

def read_log(log_path:str) -> tuple[np.ndarray,np.ndarray]:
    """read log
    
    Args
    ---------
    log_path : str
        path of optimization log 
    
    Return
    ---------
    flist : np.ndarray
        cost list, with shape (number_of_points,1)
    x_vec : np.ndarray
        parameters list, with shape (number_of_points, parameters_dim)
    """
    def converter(s):
        s = s[1:-2]
        # Split the string into individual numbers and convert them to floats
        return np.array([float(x) for x in s.split(',')])

    print("logs : \n")
    head_numbers = 0
    with open(log_path, 'r', encoding='utf-8') as file:
        for currentline, line in enumerate(file, start=0):  # count from line 1
            f_msgs = line.strip()
            print(f_msgs)
            head_numbers += 1
            if f_msgs == "##":
                break
    
    data_list = np.loadtxt(log_path,skiprows = head_numbers,usecols=(2),converters = converter,delimiter=' ',dtype = object)
    value_list = np.loadtxt(log_path,skiprows = head_numbers,usecols=(3))
    
    x_list = np.array([data_list[0]])
    for i in range(1,len(data_list)):
        x_list = np.vstack((x_list,data_list[i]))
    
    ## handling nan
    valid_flist = ~np.isnan(value_list)
    f_list = value_list[valid_flist]
    f_list = np.reshape(f_list,[-1,1])
    x_list = x_list[valid_flist.flatten(),:]
    
    return f_list, x_list

def log_visiual(path:str,visual:str = "all",extra_visual:callable = None):
    """view optimization results from log  

        Args
        ---------
        path : string
            path of optimization log 
            
        visual : str
            to choose visualization figures, should be one of 
            
                - "classic" : to view just cost and std-normalized traj
                - "advanced" : provide multidimension visualization, including : 
                    ``t-SNE`` , ``PCA`` , ``scatter matrix`` , ``parallel coordinates``
                - "all" : all of them
            
            defeault is ``"all"``
        
        extra_visual : callable
            custom defined visualization function to provide extra visualization, \
            should be a function with following args with fixed range : def extra_visual(flist,x_vec,method):...
            
                Args
                ---------
                flist : np.ndarray
                    cost list
                x_vec : np.ndarray
                    parameters list
                method : str
                    method name
    """
    f_list, x_list = read_log(path)
    
    if extra_visual is not None:
        _opt_plot(f_list,x_list,"from log : " + os.path.basename(path),visual,False)
        try:
            extra_visual(f_list,x_list,"from log : " + os.path.basename(path))
        except:...
        plt.show()
    else:
        _opt_plot(f_list,x_list,"from log : " + os.path.basename(path),visual,True)

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
            bad = False # True represent bad
            for _ in range(int(ave_times)):
                f_dict = func(x,*args,**kwargs)
                cost = np.hstack((cost,f_dict["cost"]))
                uncer = np.hstack((uncer,f_dict["uncer"]))
                try:
                    bad *= not f_dict["bad"]
                except:
                    bad = True
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

class _optimize_Exception(Exception):
    def __init__(self,err_msg:str) -> Exception:
        """cls to handle exceptions
        
        Args
        ---------
        err_msg : str 
            error msgs
        """
        Exception.__init__(self,"optimize error : " + err_msg)
        
    @staticmethod
    def err_dict(err_:str) -> str:
        """get err_msg corresponds to err_key

        Args
        ---------
        err_ : str
            error_msg or key of error_dict
            
        """
        error_dict = dict(
            nan = "func return nan",
            not_def = "not define",
            not_dict = "func should return cost dict : {cost,uncer,bad}"
        )
        return error_dict[err_] if err_ in error_dict else err_

class _opt_agent_model:
    def __init__(self,model:str = "gaussian_process",extra_dict:dict = {}):
        from agent_model import gaussian_process, neural_net
        """load agent model

        Args
        ---------
        load_path : str
            path of agent model to load
        """
        if model == "gaussian_process":
            self._agent_model = gaussian_process(**extra_dict)
        elif model == "neural_net":
            self._agent_model = neural_net(**extra_dict)
        else:
            raise _optimize_Exception("not_def")
        
    def train(self,X_train:np.ndarray,Y_train:np.ndarray):
        """train the agent model
        
        Args
        ---------
        X_train : np.ndarray
            training points, must be (n,m) like, where n is the number of training points and m is the dimension of parameters
        
        Y_train : np.ndarray
            training function values, must be (n,) like, where n is the number of training points
        
        """
        self._agent_model.model.fit(X_train,Y_train)
    
    def predict(self,X_predict:np.ndarray,return_std:bool = False,return_cov:bool = False) -> np.ndarray:
        """predict the function value
        
        Args
        ---------
        X_predict : np.ndarray
            predict points, must be (n,m) like, where n is the number of predict points, and m is the dimension of variable
        
        Returns
        ---------
        float
            predict function value
        """

        return self._agent_model.model.predict(X_predict, return_std = return_std, return_cov = return_cov)

    def save(self,pkl_path:str = None):
        """save agent model

        Args
        ---------
        pkl_path : (str, optional)
            path to save, should be .pkl file name, defeault is /opt_agent_model/...
        """
        self._agent_model.save(pkl_path)
        
    def load(self,pkl_path:str = None):
        """load the model from path

        Args
        ---------
        pkl_path : str
            path to load the model, should be .pkl file name
        """
        self._agent_model.load(pkl_path)
        
    def train_from_log(self,log_path:str):
        """train the model from log data

        Args
        ---------
        log_path : str
            path of log, should be opt_logs
        """
        flist, xlist = read_log(log_path)
        self._agent_model.train(xlist,flist)
        return self

class _opt_plot:
    def __init__(self,flist,x_vec,method,visual = "all",_show_opc:bool = True):
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
            print("classic")
            plt.figure(998,figsize=(13, 6))
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
                normal = np.std(plot_vec) + abs(mean[i])*1e-10      
                plot_vec = (plot_vec - mean[i])/normal
                plt.scatter(timelist,plot_vec,label = f"times vs paras-{i} with : [amp-std = {normal:.4f} , mean = {mean[i]:.3f}]")
            plt.legend()
            plt.xlabel("rounds")
            plt.title("std normalized parameters  @ " + str(method))
    
        # advanced : higher dimensional visualizing
        if "advanced" in visual:
            from multiprocessing import Pool
            os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Use logical cores
            from pandas import DataFrame
            from pandas.plotting import parallel_coordinates
            
            # data 
            df = DataFrame(x_vec,columns=[f"x{i}" for i in range(x_vec.shape[1])])
            df['cost'] = flist
            df['iter'] = range(1, len(df) + 1)  # Ensure 'iter' column is added
            flist = np.reshape(flist,[-1,1])
            
            with Pool(4) as pool:
                future_pc = pool.apply_async(func = _opt_plot._pc, args = (flist, x_vec,))
                future_pca = pool.apply_async(func = _opt_plot._PCA, args = (df, x_vec,))
                future_tsne = pool.apply_async(func = _opt_plot._t_SNE, args = (df, x_vec,))
                
                fig1 = future_pca.get()
                fig2 = future_tsne.get()
                df_scalar = future_pc.get()

            fig1.show()
            fig2.show()
            
            plt.figure(999, figsize=(6,6))
            parallel_coordinates(df_scalar, 'cost_range', colormap="plasma")
            plt.title('Parallel Coordinates')
            plt.xlabel('ordered parameters [0-255] normalized')
            plt.ylabel('cost [0-255] normalized')
            
            print("scatter matrix")
            from seaborn import pairplot
            grid = pairplot(df, vars=[f"x{i}" for i in range(x_vec.shape[1])],
                        hue="cost", palette='viridis', diag_kind='hist',
                        plot_kws={'alpha': 0.8}, height=8.1 / (x_vec.shape[1] + 0.5))
            plt.title("scatter matrix")
            
        plt.show(block=_show_opc)
    
    @staticmethod
    def _PCA(df,x_vec):
        print("PCA")
        from plotly.express import scatter_3d
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(x_vec)
        
        df["PCA1"] = data_pca[:,0]
        df["PCA2"] = data_pca[:,1]
        df["PCA3"] = data_pca[:,2]

        # pca results
        variance_ratio = pca.explained_variance_ratio_
        comp = np.round(pca.components_,2)
        
        fig1 = scatter_3d(df,x = "PCA1",y = "PCA2",z = "PCA3",color = "cost",size="iter",
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
        return fig1
        
    @staticmethod
    def _t_SNE(df,x_vec):
        print("t_SNE")
        from plotly.express import scatter_3d
        from sklearn.manifold import TSNE
        
        tsne = TSNE(n_components = 3,perplexity=np.min([16 , 4 + 2*x_vec.shape[1],2 + x_vec.shape[0]//50]),max_iter = 1000)
        data_tsne = tsne.fit_transform(x_vec)
        
        df["tsne1"] = data_tsne[:,0]
        df["tsne2"] = data_tsne[:,1]
        df["tsne3"] = data_tsne[:,2]
            
        fig2 = scatter_3d(df,x = "tsne1",y = "tsne2",z = "tsne3",color = "cost",size="iter",
                                title = "high dimension visual @ TSNE , focus on ** Clusters **",
                                labels = {"cost":"cost"},hover_data = [f"x{i}" for i in range(x_vec.shape[1])])
        fig2.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey',colorscale='viridis')),opacity=0.8)
        return fig2
    
    @staticmethod
    def _pc(flist,x_vec):
        print("parallel coordinate")
        from sklearn.preprocessing import MinMaxScaler
        from pandas import DataFrame, qcut
        
        scalar = MinMaxScaler(feature_range=(0,255))
        x_vec_scalar = scalar.fit_transform(x_vec)
        flist_scalar = scalar.fit_transform(flist)
        df_scalar = DataFrame(x_vec_scalar,columns=[f"x{i}" for i in range(x_vec.shape[1])])
        df_scalar["cost"] = flist_scalar
        df_scalar['cost_range'] = qcut(df_scalar['cost'], q = 15, labels = False, duplicates='drop')
        return df_scalar

class base_optimizer:
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
        self._err_msg = None
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
            ## store np/th result in list
            self._paras_init = paras_init
            self._flist = []
            self._x_vec = []
            
            self._time_stamp = [time.strftime("%d:%H:%M:%S",time.gmtime(self._time_start))]
            log_head_inhert = ""
            self._filename = kwargs.get("logfile","optimization__" + time.strftime("%Y-%m-%d-%H-%M",time.gmtime(self._time_start)) + "__" + kwargs.get("method","None") + "__" + ".txt")
            self._ave_dict = kwargs.get("ave_dict",{"ave":False,"ave_times":1,"ave_wait":0.})
            self._run_count = 0
            
        ## create log head
        if self._log or self._log == "inherit":
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
        if self._ave_dict.get("ave",False) and self._torch == False:
            func = _ave_decorate(func,self._ave_dict.get("ave_times",3),self._ave_dict.get("ave_wait",0.01))
        
        ## normal decorate
        self._func = self._decorate(func,delay = delay,msg = msg)
    
    def _logging(self,err_msg:str = ""):
        """generating loggings
        """
        
        # if there are err_msg , we will add special head !!_ in log
        self._filename = "err_" + self._filename if err_msg != "" else self._filename
            
        self._time_end = local_time()
        delta_t = self._time_end - self._time_start
        f_delta_t = time.strftime("%H:%M:%S",time.gmtime(delta_t))
        print("\nthe optimization progress costs:")
        print(f"hh:mm:ss = {f_delta_t}\n")
        
        if self._log == True or err_msg != "": ## log == "inherit" -> not writting log
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
            if type(self._flist) == list:
                if self._torch :
                    self._flist = th.stack(self._flist).cpu().detach().numpy()
                    self._x_vec = th.stack(self._x_vec).cpu().detach().numpy()
                else:
                    self._flist = np.array(self._flist)
                    self._x_vec = np.array(self._x_vec)
            with open(self._filename, "a") as file:
                for i in range(self._flist.size):
                    file.write(f"{i}" + ", " +
                                self._time_stamp[i] 
                                + ", ")
                    file.write("[" + ",".join(map(str,self._x_vec[i])) + "]")
                    file.write(", " + f"{self._flist[i]}" + "\n")
    
    def _decorate(self,func,delay = 0.1,msg = True):
        ## decorate optimization function:
        # msg -> message each call
        # delay -> delay each call
        # _torch -> whether torch func
        _isnan, eval_str = (th.isnan, "x.clone()") if self._torch else (np.isnan, "x.copy()")
        def func_decorate(x,*args,**kwargs):
            time.sleep(delay)
            f = func(x,*args,**kwargs)
            f_val = f.get("cost")
            if _isnan(f_val): # nan error
                self.error("nan") 
            print(f"INFO RUN: {self._run_count}")
            if msg:
                print(f"INFO cost {f_val:.6f}")
                print(f"INFO parameters {x}" + "\n")
            self._run_count += 1
            ## build flist including f values
            ## and x_vec in which x_vec[:,i] include the 
            self._flist.append(f_val)
            self._x_vec.append(eval(eval_str))
            self._time_stamp = self._time_stamp + [ time.strftime("%d:%H:%M:%S",time.gmtime(local_time())) ]
            return (f_val if self._val_only else f)
            
        return func_decorate
    
    def optimization(self): self.error("method has not been defined for thhis optimizer")

    def visualization(self,visual:str = "all",extra_visual:callable = None):
        """to visualize optimization results
        
            Args
            ---------
            visual : str
                to choose visualization figures, should be one of 
                
                    - "classic" : to view just cost and std-normalized traj
                    - "advanced" : provide multidimension visualization, including : 
                        ``t-SNE`` , ``PCA`` , ``scatter matrix`` , ``parallel coordinates``
                    - "all" : all of them
                
                defeault is ``"all"``
            
            extra_visual : callable
                custom defined visualization function to provide extra visualization, \
                should be a function with following args with fixed range : def extra_visual(flist,x_vec,method):...
                
                    Args
                    ---------
                    flist : np.ndarray
                        cost list
                    x_vec : np.ndarray
                        parameters list
                    method : str
                        method name
        """ 
        if type(self._flist) == list:
            if self._torch:
                self._flist = th.stack(self._flist).cpu().detach().numpy()
                self._x_vec = th.stack(self._x_vec).cpu().detach().numpy()
            else:
                self._flist = np.array(self._flist)
                self._x_vec = np.array(self._x_vec)
        
        if extra_visual is not None:
            _opt_plot(self._flist,self._x_vec,self._method,visual)
            try:
                extra_visual(self._flist,self._x_vec,self._method)
            except:...
            plt.show()
        else:
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
        self._err_msg = _optimize_Exception.err_dict(err)
        try:
            self._logging(self._err_msg)
        except:...
        raise _optimize_Exception(self._err_msg)

    def __exit__(self):
        self.__del__()
        
    def __del__(self):
        if self._err_msg is not None:
            print("\n *** optimizer got exceptions *** " + self._err_msg + " *** \n")
        else:
            print("\n *** optimizer stop working *** \n")

if __name__ == "__main__":
    def extra_vis(flist,x_vec,method):
        print("extra")
        plt.figure(1)
        plt.plot(flist,flist)
        plt.show()
    
    path = "labopt_logs/lab_opt_2025_02_19/optimization__2025-02-19-19-55__ISMA__.txt"
    log_visiual(path,visual="all",extra_visual=extra_vis)
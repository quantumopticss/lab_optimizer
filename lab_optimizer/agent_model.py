import torch as th
from torch import nn
import torch.optim as optim

import numpy as np
import joblib
import time
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared,Matern,RBF, ConstantKernel, WhiteKernel

# general agent models
# provide : gaussian process, neural net

class agent_model:
    def __init__(self):
        self._saved = None
    def __del__(self):
        if self._saved == None and self._to_save:
            self.save()
    def save(self,obj,pkl_path:str = None,decstr:str = ""):
        """save agent model

        Args
        ---------
        pkl_path : str
            path to save, should be .pkl file name, defeault is /opt_agent_model/...
            
        decstr : str    
            decorate string
        """
        if pkl_path == None:
            ## folder
            os.makedirs("opt_agent_model", exist_ok=True)
            sub_folder = os.path.join("opt_agent_model","opt_agent_model" + time.strftime("%Y_%m_%d",time.gmtime(time.time())) )
            os.makedirs(sub_folder, exist_ok=True)
            _filename = os.path.join(sub_folder, "agent_ " + decstr + "__" + time.strftime("%Y_%m_%d_%H_%M",time.gmtime(time.time())) + "__.pkl") 
            ## save
            joblib.dump(obj, _filename)
            self._saved = True
        else:
            ## save
            try:
                joblib.dump(obj, pkl_path)
                self._saved = True
            except Exception as e: 
                temp_path = "temp_agent_model" + "__" + time.strftime("%Y_%m_%d_%H_%M",time.gmtime(time.time())) + "__.pkl"
                joblib.dump(obj, temp_path)
                self._saved = True
                print(f"wrong pkl_path, temporal agent model saved in {temp_path}")
                raise e
    def load(self,pkl_path:str):
        """load the model

        Args
        ---------
        pkl_path : str
            path to load the model, should be .pkl file name
            
        Return
        ---------
        return joblib.load(pkl_path)    
        """
        try:
            return joblib.load(pkl_path)
        except Exception as e:
            print("load failed")
            raise e
    def train(self):...
    def predict(self):...

class gaussian_process(agent_model):
    def __init__(self,kernel_type = "general", noise_level = 1.0,save = True):
        """initialize a gaussian process agent model

        Args
        ---------
        kernel : str
            kernal kind, default is "general" : 
                - smooth -> radial basic function kernel
                - periodic -> periodic kernel
                - nonlinear -> matern kernel
                - general -> (smooth + nonlinear)/2
                
        noise_level : float
            expectation of the noise level value, defeault is 1.
        """
        super().__init__()
        self._to_save = save
        C = ConstantKernel(1.0, (1e-5,1e3))
        self.trained_x = np.empty((0,))
        self.trained_y = np.empty((0))
        match kernel_type:
            case "smooth":
                kernel = C*RBF(length_scale = 1.0) + WhiteKernel(noise_level=noise_level, noise_level_bounds = (1e-7,1e3))
            case "periodic":
                kernel = C*ExpSineSquared(length_scale = 1.0,periodicity = 1.0) + WhiteKernel(noise_level=noise_level, noise_level_bounds = (1e-7,1e3)) 
            case "nonlinear":
                kernel = C*Matern(length_scale = 1.0,nu = 2.5) + WhiteKernel(noise_level=noise_level, noise_level_bounds = (1e-7,1e3))
            case _:
                kernel = C*(RBF(length_scale = 1.0) + Matern(length_scale = 1.0,nu = 1.5)) + WhiteKernel(noise_level=noise_level, noise_level_bounds = (1e-7,1e3))
                
        self.model = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha = noise_level)
        
    def train(self,X_train:np.ndarray,Y_train:np.ndarray):
        """train the agent model
        
        Args
        ---------
        X_train : np.ndarray
            training points, must be (n,m) like, where n is the number of training points and m is the dimension of parameters
        
        Y_train : np.ndarray
            training function values, must be (n,) like, where n is the number of training points
        
        """
        self.trained_x = np.concatenate((self.trained_x,X_train),axis = 0)
        self.trained_y = np.concatenate((self.trained_y,Y_train),axis = 0)
        self.model.fit(X_train,Y_train)
        
    def predict(self,X_predict:np.ndarray,return_std:bool = False,return_cov:bool = False) -> np.ndarray:
        """predict the function value
        
        Args
        ---------
        X_predict : np.ndarray
            predict points, must be (n,m) like, where n is the number of predict points and m is the dimension of parameters
        
        Returns
        ---------
        Y_predict : np.ndarray
            predict function value
        std : np.ndarray
            return uncertainty, if return_std == True
        cov : np.ndarray
            return covariance matrix, if return_cov == True
        """

        return self.model.predict(X_predict, return_std = return_std, return_cov = return_cov)

    def save(self,pkl_path:str = None):
        to_save = (self.model, self.trained_x, self.trained_y)
        super().save(to_save,pkl_path,decstr="gaussian_process")

    def load(self, pkl_path: str = None):
        try:
            self.model, self.trained_x, self.trained_y = super().load(pkl_path)
        except Exception as e:
            print(f"Error loading model: {e}")

class neural_net(nn.Module,agent_model):
    def __init__(self,ndim:int = 3,hidden_units:int = 64,save:bool = True):
        """initialize a neural_net agent model

        Args
        ---------
        ndim : int
            dimension of parameters
            
        hidden_units : int
            number of hidden units for each layer in the neural network, defeault is 64
            
        save : bool
            whether to save the model, defeault is True
        """
        nn.Module.__init__(self)
        agent_model.__init__(self)
        self._to_save = save
        self.model = nn.Sequential(
            nn.Linear(ndim, hidden_units),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Dropout(p=0.18),  # Dropout layer for regularization
            nn.Linear(hidden_units, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, 1)
        )
        self.cost_func = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.02)
    def forward(self,x):
        return self.model(x)
    def train(self,X_train:np.ndarray|th.Tensor,Y_train:np.ndarray|th.Tensor):
        """train the agent model
        
        Args
        ---------
        X_train : np.ndarray | th.Tensor
            training points, must be (n,m) like, where n is the number of training points and m is the dimension of parameters
        
        Y_train : np.ndarray | th.Tensor
            training function values, must be (n,) like, where n is the number of training points
        
        """
        if isinstance(X_train, np.ndarray):
            X_train = th.tensor(X_train.copy(),dtype = th.float64,requires_grad=False)
            Y_train = th.tensor(Y_train.copy(),dtype = th.float64,requires_grad=False)
        
        n, _ = X_train.shape
        for i in range(n):
            self.optimizer.zero_grad()
            outputs = self.model(X_train[i,:])
            loss = ((outputs-Y_train[i])**2).sum()
            loss.backward()
            self.optimizer.step()
    def predict(self,X_predict:np.ndarray|th.Tensor) -> np.ndarray:
        """predict the function value
        
        Args
        ---------
        X_predict : np.ndarray | th.Tensor
            predict points, must be (n,m) like, where n is the number of predict points and m is the dimension of parameters
        
        """
        if isinstance(X_predict, np.ndarray):
            X_predict = th.tensor(X_predict.copy(),dtype = th.float64,require_grad = False)
        
        self.model.eval()  # Ensure the model is in evaluation mode
        with th.no_grad():  # No gradient calculation needed during prediction
            y_pred = self.model(X_predict).squeeze()  # Remove extra dimensions if necessary
        
        return y_pred.numpy()
    def save(self,pkl_path:str = None):
        super().save(self.model,pkl_path,decstr="neural_net")
    def load(self,pkl_path:str = None):
        self.model = super().load(pkl_path)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.005)

def main_numpy():
    import matplotlib.pyplot as plt
    from opt_lib.test_functions import F10 as FF
    test_func = lambda x: FF(x) + np.random.rand() * 0.2
    x = np.linspace(-8,8,16).reshape(-1,1)
    x += np.random.rand(*x.shape)
    y0 = np.apply_along_axis(test_func,axis = 1,arr = x)
    XX = np.linspace(-12,12,100).reshape(-1,1)
    YY = np.apply_along_axis(test_func,axis = 1,arr = XX)

    model = gaussian_process(noise_level=0.01,save = False)
    model.train(x,y0)

    x_pred = np.linspace(-15,15,200).reshape(-1,1)
    y_pred, sigma = model.predict(x_pred, return_std=True)

    plt.figure(1)
    plt.scatter(x, y0, color='r', label='Training data')
    plt.plot(x_pred, y_pred, color='b', label='Predicted function')
    plt.plot(XX, YY, color='g', label='raw function')
    plt.fill_between(x_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, color='gray', alpha=0.2, label='95% confidence interval')
    plt.legend()
    plt.show()
    
def main_torch():
    import matplotlib.pyplot as plt
    from opt_lib.test_functions import F9 as FF
    test_func = lambda x: FF(x) + np.random.rand() * 0.2
    x = th.linspace(-2,12,80).reshape(-1,1)
    y = [test_func(i) for i in x] 
    
    XX = th.linspace(-5,15,150).reshape(-1,1)
    YY = [test_func(i) for i in XX] 

    model = neural_net(ndim = 1,save = False)
    model.train(x,y)

    x_pred = th.linspace(-5,15,200).reshape(-1,1)
    y_pred = model.predict(x_pred)

    plt.figure(1)
    plt.scatter(x.numpy(), y, color='r', label='Training data')
    plt.plot(x_pred.numpy(), y_pred, color='b', label='Predicted function')
    plt.plot(XX.numpy(), YY, color='g', label='raw function')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main_numpy()
    # main_torch()
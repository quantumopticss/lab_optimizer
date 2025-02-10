# ISMA algorithm (minimize func)
# ref :  https://doi.org/10.1016/j.cma.2022.115764
# ref :  Digital Object Identifier 10.1109/ACCESS.2025.3527509

import numpy as np
from scipy.optimize import minimize
from numpy.random import rand

class ISMA:
    @staticmethod
    def _doc():
        return "Improved Slime Mould Algorithm"
    
    def __init__(self,func:callable,paras_init:np.ndarray,bounds:tuple,args:tuple = (),max_run:int = 37,pop:int = 5,local_polish:bool = True,slime_paras:dict = dict(z = 0.03, a = 0, b=0.01,eps = 5e-2),**extra_dict):
        """ Multi Slime Mould Algorithm
        
        Args
        ---------
        func : callable
            function to opt
        
        paras_init : np.ndarray
            init states
            
        bounds : tuple
            boundary
            
        args : tuple
            extra arguments of func
            
        max_run : int
            maximum iterations, , defeault is 37
            
        Extra_Args
        ---------
        pop : int
            population of slime mould, defeault is 5
            
        local_polish : bool
            whether to do local_optimization where searching for global_opt during the whole iteration, defeault is True
            
        slime_paras : dict
            basic properties of slime mould :
                z : probability of nature mutation 
                a & b : parameters for searching
                eps : mutation step size 
                
            defeault is dict(z = 0.03, a = 0, b=0.01,eps = 5e-2)
        """
        
        def sign_dec(func): ## change sign
            def wrap(x,*args,**kwargs):
                f = - func(x,*args,**kwargs)
                return f
            return wrap
        
        ## initialize other parameters
        self._dim = len(paras_init)
        self._T = np.max([3*self._dim,max_run//6])
        self._t = 0
        self._max_run = max_run
        self._pop = pop
        
        self._neg_func = sign_dec(func) ## negative function -> maximize
        self._pos_func = func ## positive function -> minimize
        
        self._args = args
        self._local_polish = local_polish
        self._bounds = bounds
        
        self._eps = slime_paras.get("eps",0.05)
        self._a = slime_paras.get("a",0.)
        self._b = slime_paras.get("b",0.01)
        self._z = slime_paras.get("z",0.03)
        
        ## bounds
        ub, lb = [], []
        for i in range(self._dim):
            ub.append(bounds[i][1])
            lb.append(bounds[i][0])
        self.ub, self.lb = np.array(ub), np.array(lb)
        
        ## initialize slime population
        self._x = np.empty([self._dim,self._pop])
        self._y = np.empty([self._pop])
        
        self._x[:,0] = paras_init
        for i in range(1,self._pop):
            s = np.cos(2*np.pi*rand(self._dim))
            self._x[:,i] = 0.5*( self.ub*(1+s) + self.lb*(1-s) )

    def run(self,mutation:float = 0.01):
        """run ISMA algorithm

        Args
        ---------
        mutation : float, optional
            mutation probability in each iteration, should be in range [0,0.1], defaults to 0.01.
        """
        ## check value
        if mutation < 0. or mutation > 0.1:
            raise ValueError("muttion should be in range [0,0.1]")
        
        ## *** opt *** 
        for t in range(1,1+self._max_run):
            ## calculate value
            for i in range(self._pop):
                self._y[i] = self._neg_func(self._x[:,i],*self._args)
            
            idx_min = np.argmin(self._y)
            y_min = self._y[idx_min]
            idx_max = np.argmax(self._y)
            y_max = self._y[idx_max]
            
            ## local polish
            if self._local_polish and t%self._T == 0:
                self._x[:,idx_max], y_max = self.__polish(self._pos_func,self._x[:,idx_max],bounds = self._bounds,args = self._args)
            
            ## update
            u = np.arctanh(1 - t/(self._max_run))
            dx_pop = np.empty_like(self._x) # store x in column
            for i in range(self._pop):
                r = rand()
                p = np.tanh( np.abs( self._y[i] -  y_max)  )
                
                if rand() < self._z: ## mutation
                    idx_r1_r2 = np.random.randint(self._pop,size = 2)
                    dx_pop[:,i] = self._eps * (self._x[:,idx_r1_r2[0]] - self._x[:,idx_r1_r2[1]])
                elif r<p:
                    c1 = ( 2*rand() - 1 )*u
                    y1 = self._y[i]
                    
                    str_G = np.sum( y1 > self._y )
                    if str_G >= self._pop//2:
                        G = 1 + r*np.log( 1 + (y_max - y1)/(y_max-y_min+1e-10) )
                    else:
                        G = 1 - r*np.log( 1 + (y_max - y1)/(y_max-y_min+1e-10) )
                    
                    dx_pop[:,i] = c1 * ( G * self._x[:,idx_max] - self._x[:,i] )
                else:
                    pr = rand(self._dim)
                    dx_pop[:,i] = ( self._a + self._b*np.tan( np.pi*(pr-1/2) ) - 1 ) * self._x[:,i]
                
            self._x = self._x + dx_pop
            
            ## mutation
            if rand() <= mutation:
                num = 1 + self._dim//10
                prob = num/self._dim
                
                mutation_index = rand(self._dim) <= prob
                
                for i in range(self._pop):
                    self._x[:,i] = self._x[:,i] + self._eps*(self.ub - self.lb)*(2*rand(self._dim)-1)*mutation_index
            
            ## boundary control
            self._x = np.clip(self._x, self.lb[:, None], self.ub[:, None])
            # * equivalent to * #
            # for i in range(self._pop):
            #     f = self._x[:,i]
            #     (self._x[:,i])[f>=self.ub] = self.ub[f>=self.ub]
            #     (self._x[:,i])[f<=self.lb] = self.lb[f<=self.lb]
            
        ## *** opt result ***
        for i in range(self._pop):
            self._y[i] = self._neg_func(self._x[:,i],*self._args)
        
        idx_max = np.argmax(self._y)
        self.x = self._x[:,idx_max]
        
        self.x_optimize, self.y_optimize = self.__polish(self._pos_func,self.x,self._bounds,self._args)
        self.x = self.x_optimize
        self.y = self.y_optimize
        
        return self.x_optimize

    def __polish(self,func,paras_init,bounds,args = (),polish_method = "L-BFGS-B"): 
        """ polish opt result ( a small operate of minimizing func )
        """
        res = minimize(func,
                    np.copy(paras_init),
                    method=polish_method,
                    bounds=bounds,
                    args = args,
                    options = dict(maxiter = 3))
        
        return res.x, res.fun

def main():
    from test_functions import F7 as FF
    paras_init = np.array([99,120,-200])
    from time import sleep
    
    def f_dec(fun):
        def wrap(x,*args,**kwargs):
            sleep(0.05)
            f=fun(x,*args,**kwargs)
            print(f"val = {f}")
            print(f"x = {x} \n")
            return f
        return wrap
    
    func = f_dec(FF)
    
    bounds = ((-500,500),(-500,500),(-500,500))
    res = ISMA(func,paras_init,bounds = bounds,pop = 5,max_iter=1,local_polish=False)
    res.run()
    x_opt = res.x
    
    print("res:")
    print(x_opt)
    print(func(x_opt))

if __name__ == "__main__":
    main()
    
del main

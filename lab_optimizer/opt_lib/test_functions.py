## this py file provide standard optimization test functions

"""
x : np.ndarray|th.Tensor|float
    input of the function, with dimension (n,)
"""

import numpy as np
import torch as th
from typing import Union

def F1(x: Union[np.ndarray,th.Tensor,float]):
    r"""f = \sum_{i=1}^{dim} (x_i^2)
    """
    return (x**2).sum() if isinstance(x, (np.ndarray, th.Tensor)) else x**2

def F2(x: Union[np.ndarray, th.Tensor, float]):
    r"""f = \sum_{i=1}^{dim} |x_i| + \prod_{i=1}^{dim} |x_i|
    """
    if isinstance(x, (np.ndarray, th.Tensor)):
        return abs(x).sum() + abs(x).prod()
    
    return 2*abs(x)  

def F3(x: Union[np.ndarray, th.Tensor]):
    r"""f = \sum_{i=1}^{dim} ( \sum_{j=1}^{i} x_j )^2
    """
    if isinstance(x, th.Tensor):
        return (th.cumsum(x, dim=0) ** 2).sum()  ##
    return (np.cumsum(x) ** 2).sum() 

def F4(x: Union[np.ndarray,th.Tensor,float]):
    r"""f = max_{i} ( abs(x_i) )
    """
    if isinstance(x, (np.ndarray, th.Tensor)):
        return abs(x).max()
    
    return abs(x)

def F5(x:Union[np.ndarray,th.Tensor]):
    r"""f = \sum_{i=1}^{dim-1} ( 100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 )
    """
    x_r = x[1:]
    x_l = x[0:-1]
    f = (100*(x_r - x_l**2)**2 + (x_l - 1)**2 ).sum()
    return f

def F6(x: Union[np.ndarray, th.Tensor, float]):
    r"""f = \sum_{i=1}^{dim} ( floor(x_i + 0.5) )^2
    """
    if isinstance(x, (np.ndarray, th.Tensor)):
        rounded_x = x.floor() if isinstance(x, th.Tensor) else np.floor(x + 0.5)
        return (rounded_x ** 2).sum()
    
    return np.floor(x + 0.5) ** 2  

def F7(x: Union[np.ndarray,th.Tensor,float]):
    r"""f = \sum_{i=1}^{dim} ( i*x_i^4 )
    """
    if isinstance(x, (th.Tensor, np.ndarray)):
        if isinstance(x, th.Tensor):
            i_list = th.arange(1,len(x)+1,dtype = x.dtype, device = x.device)
        i_list = np.arange(1,len(x)+1,dtype = x.dtype, device = x.device)
        return (i_list*x**4).sum()
        
    return x**4

def F8(x: Union[np.ndarray,th.Tensor,float]):
    r"""f = \sum_{i=1}^{dim} ( -x_i sin(\sqrt{|x_i|}) )
    """
    if isinstance(x, (np.ndarray, th.Tensor)):
        sqrt_x = th.sqrt(abs(x)) if isinstance(x, th.Tensor) else np.sqrt(abs(x))
        f = - (x*sqrt_x).sum()
        return f

    return -x*np.sin(np.sqrt(np.abs(x)))

def F9(x: Union[np.ndarray,th.Tensor,float]):
    r"""f = \sum_{i=1}^{dim} ( x_i^2 - 10cos(2\PI x_i) + 10 )
    """
    f = 0.
    if isinstance(x, (np.ndarray, th.Tensor)):
        _cos = th.cos(2*th.pi*x) if isinstance(x, th.Tensor) else np.cos(2*np.pi*x)
        f = (x**2 - 10*_cos + 10).sum()
        return f 
        
    return x**2 - 10*np.cos(2*np.pi*x) + 10

def F10(x: Union[np.ndarray,th.Tensor,float]):
    r"""f = 20 + e - 20*exp( -0.2\sqrt{ 1/dim * \sum_{i=1}^{dim} x_i^2 } ) - exp(1/dim \sum_{i=1}^{dim} cos(2 \PI x_i))
    """
    ## np.e is equivalent to th.e to 2.7182818284...
    if isinstance(x, (np.ndarray, th.Tensor)):
        if isinstance(x, th.Tensor):
            _exp = th.exp
            _sqrt = th.sqrt
            _cos = th.cos
            e = th.e
        else:
            _exp = np.exp
            _sqrt = np.sqrt
            _cos = np.cos
            e = np.e
        f = 20 + e - 20*_exp(-0.2*_sqrt(1/len(x)*(x**2).sum())) - _exp(1/len(x)* (_cos(2*th.pi*x)).sum()  )
        return f
    return 20 + np.e - 20*np.exp(-0.2*abs(x) ) - np.exp(np.cos(2*np.pi*x))

def F11(x:Union[np.ndarray,th.Tensor,float]):
    r"""f = 1/4000 * \sum_{i=1}^{dim} x_i^2 - \PI_{i=1}^{dim} cos(x_i/\sqrt{i}) + 1
    """
    if isinstance(x, (th.Tensor, np.ndarray)):
        if isinstance(x, th.Tensor):
            _cos = th.cos
            _sqrt = th.sqrt
            i_list = th.arange(1,len(x)+1,dtype = x.dtype, device = x.device)
        else:
            _cos = np.cos
            _sqrt = np.sqrt
            i_list = np.arange(1,len(x)+1,dtype = x.dtype, device = x.device)
        
        f = 1/4000*(x**2).sum() - (_cos(x/_sqrt(i_list))).prod() + 1
        return f
    
    f = 1/4000*x**2 - np.cos(x) + 1
    return f

def u(x: Union[th.Tensor,np.ndarray,float],a:float,k:float,m:float):
    f = k*(x-a)**m * (x>a) + k*(-x-a)**m * (x<-a)
    return f

def y(x:float):
    r"""y = 1 + (x+1)/4
    """
    f = x/4 + 1.25
    return f

def F12(x: Union[np.ndarray,th.Tensor]):
    r"""f = \sum_{i}^{dim} u(x_i,5,100,4) + \frac{\pi}{dim} \left[ 10 \sin^2(\pi*y1) + \sum_{i=1}{dim-1} (y_i-1)**2 (1 + 10 \sin^2(\pi y_{i+1})) + (y_{dim} -1)**2 \right]
    """
    ## np.pi is equivalent to th.pi to 3.1415926...
    _sin, _pi = (th.sin, th.pi, ) if isinstance(x, th.Tensor) else (np.sin, np.pi, )
    f = u(x,5,100,4).sum() + _pi/len(x)*( 10*_sin(_pi*y(x[0]))**2 + ( (y(x[0:-1])-1)**2 * (1 + 10*_sin(_pi*y(x[1:]))**2)).sum() + (y(x[-1])-1)**2  )
    return f

def F13(x: Union[np.ndarray,th.Tensor]):
    r"""f = 0.1 * ( 
    \sin^2(3 \pi x_1) + \sum_{i=1}^{dim-1} (x_i - 1)^2 ( 1 + \sin^2(3 \pi x_{i+1}) )
    ) + \sum_{i=1}^{dim} u(x_i,5,100,4)
    """
    ## np.pi is equivalent to th.pi to 3.1415926...
    _sin, _pi = (th.sin, th.pi, ) if isinstance(x, th.Tensor) else (np.sin, np.pi, )
    f = 0.1 * ( _sin(3*_pi*x[0])**2 + ( (x[0:-1]-1)**2 * (1 + _sin(3*_pi*x[1:])**2) ).sum() ) + u(x,5,100,4).sum()
    return f

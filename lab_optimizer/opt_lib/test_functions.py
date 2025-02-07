## this py file provide standard optimization test functions

"""
x : np.ndarray|th.Tensor|float
    input of the function, with dimension (n,)
"""

import numpy as np
import torch as th

def F1(x:np.ndarray|th.Tensor|float):
    """
    f = \sum_{i=1}^{dim} (x_i^2)
    """
    if type(x) == th.Tensor:
        f = th.sum(x**2,dim = None)
    else:
        f = np.sum(x**2,axis = None) 
    return f

def F2(x:np.ndarray|th.Tensor|float):
    """
    f = \sum_{i=1}^{dim} ( abs(x_1) ) + \PI_{i=1}^{dim}( abs(x_i) )
    """
    if type(x) == th.Tensor:
        f = th.sum(th.abs(x)) + th.prod(th.abs(x))
    else:
        f = np.sum(np.abs(x)) + np.prod(np.abs(x))
    return f

def F3(x:np.ndarray|th.Tensor|float):
    """
    f = \sum_{i=1}^{dim} ( \sum_{j=1}^{i} x_j )^2
    """
    f = 0.
    if type(x) == th.Tensor:
        for i in range(len(x)):
            f = f + th.sum(x[:i+1])**2
    else:
        for i in range(len(x)):
            f = f + np.sum(x[:i+1])**2
    return f

def F4(x:np.ndarray|th.Tensor|float):
    """
    f = max_{i} ( abs(x_i) )
    """
    if type(x) == th.Tensor:
        f = th.max(th.abs(x))
    else:
        f = np.max(np.abs(x))
    return f

def F5(x:np.ndarray|th.Tensor|float):
    """
    f = \sum_{i=1}^{dim-1} ( 100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2 )
    """
    x_r = x[1:]
    x_l = x[0:-1]
    if type(x) == th.Tensor:
        f = th.sum(100*(x_r - x_l**2)**2 + (x_l - 1)**2)
    else:
        f = np.sum(100*(x_r - x_l**2)**2 + (x_l - 1)**2)
    return f

def F6(x:np.ndarray|th.Tensor|float):
    """
    f = \sum_{i=1}^{dim} ( floor(x_i +0.5) )^2
    """
    if type(x) == th.Tensor:
        f = th.sum(th.floor(x+0.5)**2)
    else:
        f = np.sum(np.floor(x+0.5)**2)
    return f

def F7(x:np.ndarray|th.Tensor|float):
    """
    f = \sum_{i=1}^{dim} ( i*x_i^4 )
    """
    if type(x) == th.Tensor:
        i_list = th.arange(1,len(x)+1)
        f = th.sum(i_list*x**4)
    else:
        i_list = np.arange(1,len(x)+1)
        f = np.sum(i_list*x**4)
    return f

def F8(x:np.ndarray|th.Tensor|float):
    """
    f = \sum_{i=1}^{dim} ( -x_i sin(\sqrt{|x_i|}) )
    """
    f = 0.
    if type(x) == th.Tensor:
        for s in x:
            f = f - s*th.sin(th.sqrt(th.abs(s)))
    else:
        for s in x:
            f = f - s*np.sin(np.sqrt(np.abs(s)))
    return f

def F9(x:np.ndarray|th.Tensor|float):
    """
    f = \sum_{i=1}^{dim} ( x_i^2 - 10cos(2\PI x_i) + 10 )
    """
    f = 0.
    if type(x) == th.Tensor:
        for s in x:
            f = f + s**2 - 10*th.cos(2*th.pi*s) + 10
    else:
        for s in x:
            f = f + s**2 - 10*np.cos(2*np.pi*s) + 10
    return f

def F10(x:np.ndarray|th.Tensor|float):
    """
    f = 20 + e - 20*exp( -0.2\sqrt{ 1/dim * \sum_{i=1}^{dim} x_i^2 } ) - exp(1/dim \sum_{i=1}^{dim} cos(2 \PI x_i))
    """
    if type(x) == th.Tensor:
        f = -20*th.exp( -0.2*th.sqrt(1/len(x)*th.sum(x**2)) ) - th.exp( 1/len(x)*th.sum(th.cos(2*th.pi*x)) ) + 20 + th.e
    else:
        f = -20*np.exp( -0.2*np.sqrt(1/len(x)*np.sum(x**2)) ) - np.exp( 1/len(x)*np.sum(np.cos(2*np.pi*x)) ) + 20 + np.e
    return f

def F11(x:np.ndarray|th.Tensor|float):
    """
    f = 1/4000 * \sum_{i=1}^{dim} x_i^2 - \PI_{i=1}^{dim} cos(x_i/\sqrt{i}) + 1
    """
    if type(x) == th.Tensor:
        i_list = th.arange(1,len(x)+1)
        f = 1/4000*th.sum(x**2) - th.prod(th.cos(x/th.sqrt(i_list))) + 1
    else:
        i_list = np.arange(1,len(x)+1)
        f = 1/4000*np.sum(x**2) - np.prod(np.cos(x/np.sqrt(i_list))) + 1
    return f

def u(x,a,k,m):
    if x > a:
        f = k*(x-a)**m
    elif x >= -a and x<= a:
        f = 0
    else:
        f = k*(-x-a)**m
    return f

def y(x):
    f = x/4 + 1.25
    return f

def F12(x:np.ndarray|th.Tensor|float):
    """
    f = \sum_{i}^{dim} u(x_i,5,100,4) + \frac{\pi}{dim} \left[ 10 \sin^2(\pi*y1) + \sum_{i=1}{dim-1} (y_i-1)**2 (1 + 10 \sin^2(\pi y_{i+1})) + (y_{dim} -1)**2 \right]
    """
    if type(x) == th.Tensor:
        f = th.sum(u(x,5,100,4)) + th.pi/len(x)*( 10*th.sin(th.pi*y(x[0]))**2 + th.sum( (y(x[0:-1])-1)**2 * (1 + 10*th.sin(th.pi*y(x[1:]))**2)) + (y(x[-1])-1)**2  )
    else:
        f = np.sum(u(x,5,100,4)) + np.pi/len(x)*( 10*np.sin(np.pi*y(x[0]))**2 + np.sum( (y(x[0:-1])-1)**2 * (1 + 10*np.sin(np.pi*y(x[1:]))**2)) + (y(x[-1])-1)**2  )
    return f

def F13(x:np.ndarray|th.Tensor|float):
    """
    f = 0.1 * ( 
    \sin^2(3 \pi x_1) + \sum_{i=1}^{dim-1} (x_i - 1)^2 ( 1 + \sin^2(3 \pi x_{i+1}) )
    ) + \sum_{i=1}^{dim} u(x_i,5,100,4)
    """
    if type(x) == th.Tensor:
        f = -0.1*( th.sin(3*th.pi*x[0])**2 + th.sum((x[0:-1] - 1)**2*(1 + th.sin(3*th.pi*x[1:])**2) ) + (x[-1] - 1)**2 ) + th.sum(u(x,5,100,4))
    else:
        f = -0.1*( np.sin(3*np.pi*x[0])**2 + np.sum((x[0:-1] - 1)**2*(1 + np.sin(3*np.pi*x[1:])**2) ) + (x[-1] - 1)**2 ) + np.sum(u(x,5,100,4))
    return f
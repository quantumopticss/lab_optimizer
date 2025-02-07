## this py file provide standard optimization test functions

import numpy as np

def F2(x:np.ndarray):
    """
    f = \sum_{i=1}^{dim} ( abs(x_1) ) + \PI_{i=1}^{dim}( abs(x_i) )
    """
    f = 0.
    g = 1.
    for i in x:
        f = f + np.abs(i)
        g = g*i
        
    return (f+np.abs(g))

def F8(x:np.ndarray):
    """
    f = \sum_{i=1}^{dim} ( -x_i sin(\sqrt{|x_i|}) )
    """
    f = 0.
    for s in x:
        f = f - s*np.sin(np.sqrt(np.abs(s)))
    return f

def F9(x:np.ndarray):
    """
    f = \sum_{i=1}^{dim} ( x_i^2 - 10cos(2\PI x_i) + 10 )
    """
    f = 0.
    for s in x:
        f = f + s**2 - 10*np.cos(2*np.pi*s) + 10
    return f

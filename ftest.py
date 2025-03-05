import numpy as np
import matplotlib.pyplot as plt

a1 = -1; a2 = 1
b1 = -1; b2 = 1

## 
num = 300

xlist = np.empty((2,num))

# ## rand
# for i in range(2):
#     r = np.random.rand(num)*2 - 1
#     xlist[i,:] = ( a1*(1-r) + a2*(1+r) )/2

## spherical rand
for i in range(2):
    r = np.cos(np.random.rand(num)*2*np.pi)
    xlist[i,:] = ( a1*(1-r) + a2*(1+r) )/2

fig, ax = plt.subplots()
ax.scatter(xlist[0,:], xlist[1,:], s=5, c='b')
ax.set(xlim=(a1, a2), xticks=np.arange(a1, a2, 10),
       ylim=(b1, b2), yticks=np.arange(b1, b2, 10))
ax.grid(color='r', linestyle='-', linewidth=2)
plt.show()
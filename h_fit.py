import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft

pi = np.pi
sqrt = lambda x: np.sqrt(x)
cos = lambda x: np.cos(x)
sin = lambda x: np.sin(x)

def gaussian_2d(xx,yy,sigma_1,sigma_2,theta):
    a = 1/(2*pi*sigma_1*sigma_2)*np.exp(-((xx*cos(theta) + yy*sin(theta))**2/(2*sigma_1**2) + (yy*cos(theta) - xx*sin(theta))**2/(2*(sigma_2**2)) ))
    return a
    
def h_gaussian(sigma_1x,sigma_1y,theta_1,sigma_2x,sigma_2y,theta_2):
    gamma_1x = 1/(sigma_1x)
    gamma_1y = 1/(sigma_1y)
    gamma_2x = 1/(sigma_2x)
    gamma_2y = 1/(sigma_2y)
    
    a = 0.5* ( cos(theta_2)**2/gamma_2x**2 + sin(theta_2)**2/gamma_2y**2 - cos(theta_1)**2/gamma_1x**2 - sin(theta_1)**2/gamma_1y**2 )
    b = 0.5* ( cos(theta_2)**2/gamma_2y**2 + sin(theta_2)**2/gamma_2x**2 - cos(theta_1)**2/gamma_1y**2 - sin(theta_1)**2/gamma_1x**2 )
    c = sin(theta_2)*cos(theta_2)*(1/gamma_2x**2 - 1/gamma_2y**2) - sin(theta_1)*cos(theta_1)*(1/gamma_1x**2 - 1/gamma_1y**2)
    
    phi = np.arctan(c/(a-b))/2
    if phi != 0.:
        A = 0.5*(a+b + c/(sin(2*phi)))
        B = 0.5*(a+b - c/(sin(2*phi)))
    else:
        A = a
        B = b
        
    sigma_A = sqrt(1/(2*A))
    sigma_B = sqrt(1/(2*B))
    
    sigma_xA = 1/(sigma_A)
    sigma_xB = 1/(sigma_B) 
    
    return sigma_xA, sigma_xB, phi

xlist = np.linspace(-2,2,80)
xx, yy = np.meshgrid(xlist,xlist)
hx = xlist[1] - xlist[0]

sigma_1x = 0.1
sigma_1y = 0.2
theta_1 = pi/3

sigma_2x = 0.32
sigma_2y = 0.3
theta_2 = pi/4

g1 = gaussian_2d(xx,yy,sigma_1x,sigma_1y,theta_1)
g2 = gaussian_2d(xx,yy,sigma_2x,sigma_2y,theta_2)

sigma_hx, sigma_hy, phi = h_gaussian(sigma_1x,sigma_1y,theta_1,sigma_2x,sigma_2y,theta_2)
h =  gaussian_2d(xx,yy,sigma_hx,sigma_hy,phi)

plt.figure(1)
plt.imshow(g1)
plt.title("g1")
plt.colorbar()

plt.figure(2)
plt.imshow(g2)
plt.title("g2")
plt.colorbar()

plt.figure(3)
plt.imshow(h)
print(np.sum(h,axis = None)*hx**2)
plt.title("h")
plt.colorbar()

f_h = fft.fftshift(fft.fft2(h),axes = (0,1))
f_g1 = fft.fftshift(fft.fft2(g1),axes = (0,1))
G = fft.ifftshift(fft.ifft2(f_h*f_g1),axes = (0,1))*hx**2

plt.figure(4)
plt.imshow(np.abs(G))
plt.colorbar()
plt.title("made g2")

plt.figure(5)
plt.imshow(np.abs(G) - g2)
plt.colorbar()
plt.title("differ")

plt.show()
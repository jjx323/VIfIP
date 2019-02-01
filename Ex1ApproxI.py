import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time as ti

from ellipticCauchyPro import *
from vbiIP import *

"""
This script used for evaluating Example 1 shown in 
B. Jin and J. Zou, Hierarchical Bayesian inference for ill-posed 
problems via variational method, Journal of Computational Physics, 
229, 2010, 7317-7343. 
"""

# specify the accurate solution 
expre = "sin(pi*x[0])*exp(pi*x[1]) + x[0] + x[1]"
# specify the true values of the Neumann boundary 
# the domain is a square q1 = {0} \times [0,1], q2 = {1} \times [0,1]
# q3 = [0,1] \times {0}  
q1_expre_n = "-(pi*cos(pi*x[0])*exp(pi*x[1]) + 1)"
q2_expre_n = "pi*cos(pi*x[0])*exp(pi*x[1]) + 1"
q3_expre_n = "-(pi*sin(pi*x[0])*exp(pi*x[1]) + 1)"
# specify the true values of the Dirichelet boundary 
# q4 = [0,1] \times {1}
q4_expre_d = "sin(pi*x[0])*exp(pi) + x[0] + 1"
# solving the forward problem
para = {'mesh_N': [100, 100], 'q1': q1_expre_n, 'q2': q2_expre_n, \
        'q3': q3_expre_n, 'q4': q4_expre_d, 'alpha': '1.0', 'f': '0.0', 'P': 2}

# specify the coordiantes of the measurement points 
mea = MeasurePoints(80)  # point number should be an even number

# generate measurement data by analytic solution
trueFun = lambda x, y: np.sin(np.pi*x)*np.exp(np.pi*y) + x + y
u_t = lambda dian: trueFun(dian[0], dian[1])
u_tm = np.array([u_t(dian) for dian in mea.points_m])
gH2 = GeneUH(para)
gH2.eva()
Uh = gH2.gene(mea.points_m)
d = u_tm - Uh
noise_level = 0.03   # set noise level to be 3%
np.random.seed(0)
noise = np.random.normal(0, 1, len(d))
d = d + noise_level*np.max(np.abs(d))*noise
sig = noise_level*np.max(np.abs(d))

# generate the forward operator H
para['mesh_N'] = [50, 50]
gH = GeneH(para)
gH.eva(101)  # set 101 basis functions 
H = gH.gene(mea.points_m)

# generate the regularize matrix
r, c = np.shape(H)
W = geneL(c)
# init parameters
para1 = {'alpha0': 1, 'beta0': 1e-3, 'alpha1': 1, 'beta1': 1e-10}
t0 = ti.time()
mE, covE, eta, lan, tau, ite = approxIGaussian(H, W, d, para1)
t1 = ti.time()
print('Inversion consumes ', t1-t0, 's')
print('approxI iterate ', ite, ' times')
print('The regularization parameter is ', eta[-1])

'''
post process
'''
# calculate estimated function 
xx = np.linspace(0, 1, 100)
yy = xx*0.0 + 1.0
gH.setBasis(mE)
fmE = gH.calBasisFun(xx)
fm = trueFun(xx, yy)

res_opt = np.linalg.norm(fmE - fm, ord=2)/np.linalg.norm(fm, ord=2)
print('L2 norm of residual = ', res_opt*100, '%')

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(xx, fmE, color='blue', label='Estimated')
plt.plot(xx, fm, color='red', linestyle='dashed', label='True')
plt.xlabel('x coordiante')
plt.ylabel('function values')
plt.title('True and estimated functions')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(eta, '-.'), plt.plot(eta, 'o')
#plt.plot(lan, '--')
#plt.plot(tau, '--')
plt.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#xplt = np.linspace(0, 1, covE.shape[0])
#yplt = np.linspace(0, 1, covE.shape[1])
#Xplt, Yplt = np.meshgrid(xplt, yplt)
#ax.plot_surface(Xplt, Yplt, np.mat(covE).I, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#ax.set_title('Covariance')
#plt.show()



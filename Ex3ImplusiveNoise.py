import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time as ti

from ellipticCauchyPro import *
from vbiIP import *
from addNoise import *

"""
This script used for evaluating Example 1 shown in 
B. Jin, A variational Bayesian method to inverse problems with implusive noise, 
Journal of Computational Physics, 
231, 2012, 423-435. 
"""

# specify the true values of the Neumann boundary 
# the domain is a square q1 = {0} \times [0,1], q2 = {1} \times [0,1]
# q3 = [0,1] \times {0}  
q1_expre_n = "-(pi*cos(pi*x[0])*exp(pi*x[1]) + 1)"
q2_expre_n = "pi*cos(pi*x[0])*exp(pi*x[1]) + 1"
q3_expre_n = "-(pi*sin(pi*x[0])*exp(pi*x[1]) + 1)"
# specify the true values of the Dirichelet boundary 
# q4 = [0,1] \times {1}
#q4_expre_d = "sin(pi*x[0])*exp(pi) + x[0] + 1"
#q4_expre_d = "0.3<=x[0] && x[0]<=0.7 ? 0.5 : 0"
q4_expre_d = "0.0<=x[0] && x[0]<=0.5 ? 2*x[0] : (0.5<=x[0] && x[0]<=1.0 ? 2-2*x[0] : 0)"
# solving the forward problem
para = {'mesh_N': [100, 100], 'q1': q1_expre_n, 'q2': q2_expre_n, \
        'q3': q3_expre_n, 'q4': q4_expre_d, 'alpha': '1.0', \
        'f': '0.0', 'P': 2}

# specify the coordiantes of the measurement points 
mea = MeasurePoints(80)  # point number should be an even number

# generate measurement data by employing FEM on a fine mesh
fineFun = calTrueSol(para)
u_t = lambda dian: fineFun(dian[0], dian[1])
u_tm = np.array([u_t(dian) for dian in mea.points_m])
gH2 = GeneUH(para)
gH2.eva()
Uh = gH2.gene(mea.points_m)
paraNoise = {'rate': 0.5, 'noise_level': 2}
u_tm_n, sig = addGaussianNoise(u_tm, paraNoise, 'y')
d = u_tm_n - Uh

# generate the forward operator H
para['mesh_N'] = [50, 50]
gH = GeneH(para)
gH.eva(101)  # set 101 basis functions 
#H = gH.gene(mea.points_m)
#np.save('Hsave.npy', H)
H = np.load('Hsave.npy')

# generate the regularize matrix
r, c = np.shape(H)
W = geneL(c)
# Alg I with Gaussian noise assumption
para1 = {'alpha0': 1, 'beta0': 1e-3, 'alpha1': 1, 'beta1': 1e-7}
t01 = ti.time()
mE1, precisionMatrix1, eta1, lan1, tau1, ite1 = approxIGaussian(H, W, d, para1)
t11 = ti.time()
# Alg II with Gaussian noise assumption
para2 = {'alpha0': 1, 'beta0': 1e-3, 'alpha1': 1, 'beta1': 1e-7}
t02 = ti.time()
mE2, eta2, lan2, tau2, ite2 = approxIIGaussian(H, W, d, para1)
t12 = ti.time()
# Alg I with centered t-distribution noise assumption
para3 = {'alpha0': 1, 'beta0': 1e-3, 'alpha1': 1, 'beta1': 1e-7}
t03 = ti.time()
mE3, precisionMatrix3, lan3, error, wS, ite3 = approxICenteredT(H, W, d, para2)
t13 = ti.time()

'''
post process
'''
# calculate the estimated function 
xx = np.linspace(0, 1, 100)
yy = xx*0.0 + 1.0
gH.setBasis(mE1)
fmE1 = gH.calBasisFun(xx)
gH.setBasis(mE2)
fmE2 = gH.calBasisFun(xx)
gH.setBasis(mE3)
fmE3 = gH.calBasisFun(xx)
# calculate the true function
trueFun = fe.Expression(q4_expre_d, degree=5)
trueFun = np.vectorize(trueFun)
fm = trueFun(xx, yy)

print('Approx I:')
print('Inversion consumes ', t11-t01, 's')
print('approxI iterate ', ite1, ' times')
print('The regularization parameter is ', eta1[-1])
res_opt1 = np.linalg.norm(fmE1 - fm, ord=2)/np.linalg.norm(fm, ord=2)
print('L2 norm of residual = ', res_opt1*100, '%')
print('')
print('Approx II:')
print('Inversion consumes ', t12-t02, 's')
print('approxI iterate ', ite2, ' times')
print('The regularization parameter is ', eta2[-1])
res_opt2 = np.linalg.norm(fmE2 - fm, ord=2)/np.linalg.norm(fm, ord=2)
print('L2 norm of residual = ', res_opt2*100, '%')
print('')
print('Roboust Approx I:')
print('Inversion consumes ', t13-t03, 's')
print('approxII iterate ', ite3, ' times')
print('The regularization parameter is ', lan3[-1])
res_opt3 = np.linalg.norm(fmE3 - fm, ord=2)/np.linalg.norm(fm, ord=2)
print('L2 norm of residual = ', res_opt3*100, '%')

plt.figure()
plt.plot(xx, fmE1, color='red', label='Estimated by Alg I')
plt.plot(xx, fmE2, color='black', linestyle='dashed', label='Estimated by Alg II')
plt.plot(xx, fmE3, color='red', linestyle='dashed', label='Estimated by robust Alg I')
plt.plot(xx, fm, color='blue', label='True')
plt.xlabel('x coordiante')
plt.ylabel('function values')
plt.title('True and estimated functions')
plt.legend()
plt.show()

plt.figure()
plt.plot(wS/np.max(wS), 'o-')
plt.plot(d/np.max(d), '*-')
plt.show()

plt.figure()
plt.plot(d/np.max(d))
dture = u_tm - Uh
plt.plot(dture/np.max(dture))
plt.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#xplt = np.linspace(0, 1, precisionMatrix2.shape[0])
#yplt = np.linspace(0, 1, precisionMatrix2.shape[1])
#Xplt, Yplt = np.meshgrid(xplt, yplt)
#ax.plot_surface(Xplt, Yplt, np.mat(precisionMatrix2).I, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#ax.set_title('Covariance')
#plt.show()



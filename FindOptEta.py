import numpy as np
import fenics as fe
import matplotlib.pyplot as plt
import time as ti
from generateH import *
from geneOP import *


# specify the accurate solution 
expre = "sin(pi*x[0])*exp(pi*x[1]) + x[0] + x[1]"
# the domain is a square q1 = {0} \times [0,1], q2 = {1} \times [0,1]
# q3 = [0,1] \times {0}  
q1_expre_n = "-(pi*cos(pi*x[0])*exp(pi*x[1]) + 1)"
q2_expre_n = "pi*cos(pi*x[0])*exp(pi*x[1]) + 1"
q3_expre_n = "-(pi*sin(pi*x[0])*exp(pi*x[1]) + 1)"
# specify the true values of the Dirichelet boundary 
q4_expre_d = "sin(pi*x[0])*exp(pi) + x[0] + 1"
# solving the forward problem
para = {'mesh_N': [100, 100], 'q1': q1_expre_n, 'q2': q2_expre_n, \
        'q3': q3_expre_n, 'q4': q4_expre_d, 'alpha': '1.0', 'f': '0.0', 'P': 2}

# specify the coordiantes of the measurement points 
mea = MeasurePoints(80)  # point number should be an even number

# generate measurement data by analytic solution
u_t = lambda dian: trueFun(dian[0], dian[1])
u_tm = np.array([u_t(dian) for dian in mea.points_m])
gH2 = GeneUH(para)
gH2.eva()
Uh = gH2.gene(mea.points_m)
d = u_tm - Uh
noise_level = 0.03
np.random.seed(0)
noise = np.random.normal(0, 1, len(d))
d = d + noise_level*np.max(np.abs(d))*noise  #  inconsistent need to be noticed

# generate the forward operator H
para['mesh_N'] = [50, 50]
gH = GeneH(para)
gH.eva(101)  # set 101 basis functions 
H = gH.gene(mea.points_m)
r, c = np.shape(H)
#m0 = np.zeros((c, 1))
W = geneL(c)

# find the optimal regularization parameter
xx = np.linspace(0, 1, 100)
yy = xx*0.0 + 1.0
fm = trueFun(xx, yy)
# 200 uniformly distributed values for \eta in a logaritmic scale in the interval [10^{-16}, 1]
etas = np.power(10, np.linspace(-16, np.log10(1), 200))
res = []
for eta in etas:
    mE = findMinL2(H, W, d, eta)
    gH.setBasis(mE)
    fmE = gH.calBasisFun(xx)
    res.append(np.linalg.norm(fmE - fm, ord=2)/np.linalg.norm(fm, ord=2))

in_opt = np.argmin(res)
eta_opt = etas[in_opt]
print('The optimal regularization parameter eta = ', eta_opt)

t0 = ti.time()
mE = findMinL2(H, W, d, eta_opt)
t1 = ti.time()
print('Inversion consumes ', t1 - t0, 's')

'''
post process
'''
# calculate estimated function 
xx = np.linspace(0, 1, 100)
yy = xx*0.0 + 1.0
gH.setBasis(mE)
fmE = gH.calBasisFun(xx)
fm = trueFun(xx, yy)

# show results
#res_opt = np.linalg.norm((fmE - fm)/np.max(fm), ord=np.inf)
#print('L^infity norm of residual = ', res_opt*100, '%')
res_opt = np.linalg.norm(fmE - fm, ord=2)/np.linalg.norm(fm, ord=2)
print('L2 norm of residual = ', res_opt*100, '%')

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(xx, fmE, color='blue', label='Estimated')
plt.plot(xx, fm, color='red', linestyle='dashed', label='True')
plt.xlabel('x coordiante')
plt.ylabel('function values')
plt.title('True and estimated functions')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(np.linspace(1, len(res), len(res)), np.log(res))
plt.plot(in_opt, np.log(res[in_opt]), 'o', markersize=10)
plt.show()


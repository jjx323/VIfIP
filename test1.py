import numpy as np 
import matplotlib.pyplot as plt 
import fenics as fe 

from HelmholtzPro import *

plt.close()

q_fun = fe.Expression('1.0+((0.5 <= x[0] && x[0] <= 1.5 && 0.5 <= x[1] && x[1] <= 1.5) ? 1 : 0)',
                      degree=3)

domain_para = {'nx': 300, 'ny': 300, 'dPML': 0.15, 'xx': 2.0, 'yy': 2.0, 'sig0': 1.5, 'p': 2.3}
equ_para = {'kappa': 20}

do = Domain(domain_para)
hel = Helmholtz(do, equ_para)
hel.geneForwardMatrix(q_fun)
hel.addPointSource()
hel.solve()

uR1 = hel.uReal
uI1 = hel.uImag

do.modifyNxNy(200, 200)
hel.modifyDomain(do)
hel.geneForwardMatrix(q_fun)
hel.addPointSource()
hel.solve()

uR2 = hel.uReal
uI2 = hel.uImag

# compare different solutions
S = fe.FunctionSpace(do.mesh, 'P', 2)
uR1 = fe.interpolate(uR1, S)
uR2 = fe.interpolate(uR2, S)
r1 = (0.5*fe.dot(uR1 - uR2, uR1 - uR2)*fe.dx)
r2 = (0.5*fe.dot(uR1, uR1)*fe.dx)
fenzi = fe.assemble(r1)
fenmu = fe.assemble(r2)
error = fenzi/fenmu
print(error*100)



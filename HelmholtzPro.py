import numpy as np 
import matplotlib.pyplot as plt 
import fenics as fe

"""
Solve Helmholtz equation
"""

class Domain(object):
    def __init__(self, para = {'nx': 100, 'ny': 100, 'dPML': 0.15, \
                               'xx': 2.0, 'yy': 2.0, 'sig0': 1.5, 'p': 2.3}):
        self.nx, self.ny = para['nx'], para['ny']
        self.dPML = para['dPML']
        self.sig0, self.p = para['sig0'], para['p']
        self.xx, self.yy = para['xx'], para['yy']
        self.haveMesh = False
        
    def geneMesh(self):
        dPML, xx, yy, Nx, Ny = self.dPML, self.xx, self.yy, self.nx, self.ny
        self.mesh = fe.RectangleMesh(fe.Point(-dPML, -dPML), fe.Point(xx+dPML, yy+dPML), Nx, Ny)
        self.haveMesh = True
        
    def modifyNxNy(self, nx, ny):
        self.nx, self.ny = nx, ny
        self.haveMesh = False
        
    def numberOfMesh(self):
        if self.haveMesh == False:
            print('Mesh has not been generated!')
            return 0
        else:
            return self.nx*self.ny*2
    
    def sizeOfMesh(self):
        if self.haveMesh == False:
            print('Mesh has not been generated!')
            return 0
        else:
            xlen, ylen = self.xx/self.nx, self.yy/self.ny
            return [xlen, ylen, np.sqrt(xlen**2 + ylen**2), 0.5*xlen*ylen]
        

class Helmholtz(object):
    def __init__(self, domain, para={'kappa': 5.0}):
        self.domain = domain
        self.kappa = para['kappa']
        self.haveFunctionSpace = False
        
    def modifyDomain(self, domain):
        self.domain = domain
        self.haveFunctionSpace = False
        
    def geneFunctionSpace(self):
        P2 = fe.FiniteElement('P', fe.triangle, 2)
        element = fe.MixedElement([P2, P2])
        if self.domain.haveMesh == False:
            self.domain.geneMesh()
        self.V = fe.FunctionSpace(self.domain.mesh, element)
        self.haveFunctionSpace = True
    
    def geneForwardMatrix(self, q_fun=fe.Constant(0.0), fR=fe.Constant(0.0), \
                          fI=fe.Constant(0.0)):
        if self.haveFunctionSpace == False:
            self.geneFunctionSpace()
            
        xx, yy, dPML, sig0_, p_ = self.domain.xx, self.domain.yy, self.domain.dPML,\
                                  self.domain.sig0, self.domain.p
        # define the coefficents induced by PML
        sig1 = fe.Expression('x[0] > x1 && x[0] < x1 + dd ? sig0*pow((x[0]-x1)/dd, p) : (x[0] < 0 && x[0] > -dd ? sig0*pow((-x[0])/dd, p) : 0)', 
                     degree=3, x1=xx, dd=dPML, sig0=sig0_, p=p_)
        sig2 = fe.Expression('x[1] > x2 && x[1] < x2 + dd ? sig0*pow((x[1]-x2)/dd, p) : (x[1] < 0 && x[1] > -dd ? sig0*pow((-x[1])/dd, p) : 0)', 
                     degree=3, x2=yy, dd=dPML, sig0=sig0_, p=p_)
        
        sR = fe.as_matrix([[(1+sig1*sig2)/(1+sig1*sig1), 0.0], [0.0, (1+sig1*sig2)/(1+sig2*sig2)]])
        sI = fe.as_matrix([[(sig2-sig1)/(1+sig1*sig1), 0.0], [0.0, (sig1-sig2)/(1+sig2*sig2)]])
        cR = 1 - sig1*sig2
        cI = sig1 + sig2
        
        # define the coefficients with physical meaning
        angl_fre = self.kappa*np.pi
        angl_fre2 = fe.Constant(angl_fre*angl_fre)
        
        # define equations
        u_ = fe.TestFunction(self.V)
        du = fe.TrialFunction(self.V)
        
        u_R, u_I = fe.split(u_)
        duR, duI = fe.split(du)
        
        def sigR(v):
            return fe.dot(sR, fe.nabla_grad(v))
        def sigI(v):
            return fe.dot(sI, fe.nabla_grad(v))
        
        F1 = - fe.inner(sigR(duR)-sigI(duI), fe.nabla_grad(u_R))*(fe.dx) \
            - fe.inner(sigR(duI)+sigI(duR), fe.nabla_grad(u_I))*(fe.dx) \
            - fR*u_R*(fe.dx) - fI*u_I*(fe.dx)
        
        a2 = fe.inner(angl_fre2*q_fun*(cR*duR-cI*duI), u_R)*(fe.dx) \
             + fe.inner(angl_fre2*q_fun*(cR*duI+cI*duR), u_I)*(fe.dx) \
        
        # define boundary conditions
        def boundary(x, on_boundary):
            return on_boundary
        
        bc = [fe.DirichletBC(self.V.sub(0), fe.Constant(0.0), boundary), \
              fe.DirichletBC(self.V.sub(1), fe.Constant(0.0), boundary)]
        
        a1, L1 = fe.lhs(F1), fe.rhs(F1)
        self.u = fe.Function(self.V)
        self.A1 = fe.assemble(a1)
        self.b1 = fe.assemble(L1)
        self.A2 = fe.assemble(a2)
        bc[0].apply(self.A1, self.b1)
        bc[1].apply(self.A1, self.b1)
        bc[0].apply(self.A2)
        bc[1].apply(self.A2)
        self.A = self.A1 + self.A2
        
    def addPointSource(self, points=[(1, 2)], magnitude=[1]):
        if len(points) != len(magnitude):
            print('The length of points and magnitude must be equal!')
            
        for i in range(len(magnitude)):
            fe.PointSource(self.V.sub(0), fe.Point(points[i]), magnitude[i]).apply(self.b1)
    
    def solve(self):
        self.u = fe.Function(self.V)
        fe.solve(self.A, self.u.vector(), self.b1, 'mumps')
        self.uReal, self.uImag = self.u.split()

    def drawSolution(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        imuR = fe.plot(self.uReal, title='Real part of the solution')
        plt.colorbar(imuR)
        plt.subplot(1, 2, 2)
        imuI = fe.plot(self.uImag, title='Imaginary part of the solution')
        plt.colorbar(imuI)
        plt.show()




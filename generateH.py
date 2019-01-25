import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
#from multiprocessing.dummy import Pool as ThreadPool

# used for generate H
class GeneH(object):  
    def __init__(self, para):
        # para = [nx, ny, FunctionSpace, degree, ...]
        self.nx, self.ny = para['mesh_N'][0], para['mesh_N'][1]
        self.mesh = fe.UnitSquareMesh(self.nx, self.ny)
        self.Vu = fe.FunctionSpace(self.mesh, 'P', para['P'])
        self.Vc = fe.FunctionSpace(self.mesh, 'P', para['P'])
        self.sol = []
        self.al = fe.Constant(1.0)
        self.f = fe.Constant(0.0)
        self.q = fe.Constant(0.0)
        self.theta = []
        self.mE = 0
         
    def eva(self, num):
        # construct basis functions, Set num points corresponding to num basis functions
        basPoints = np.linspace(0, 1, num)
        dx = basPoints[1] - basPoints[0]
        aa, bb, cc = -dx, 0.0, dx
        for x_p in basPoints:
            self.theta.append(fe.interpolate(fe.Expression(
                    'x[0] < a || x[0] > c ? 0 : (x[0] >=a && x[0] <= b ? (x[0]-a)/(b-a) : 1-(x[0]-b)/(c-b))', 
                    degree=2, a=aa, b=bb, c=cc), self.Vc))
            aa, bb, cc = aa+dx, bb+dx, cc+dx
            
        u_trial, u_test = fe.TrialFunction(self.Vu), fe.TestFunction(self.Vu)
        left = fe.inner(self.al*fe.nabla_grad(u_trial), fe.nabla_grad(u_test))*fe.dx
        right = self.f*u_test*fe.dx
        
        def boundaryD(x, on_boundary):
            return on_boundary and fe.near(x[1], 1.0)
        
#        def process(i):
#            uH = fe.Function(self.Vu)
#            bcD = fe.DirichletBC(self.Vu, self.theta[i], boundaryD)
#            left_m, right_m = fe.assemble_system(left, right, bcD)
#            fe.solve(left_m, uH.vector(), right_m)
#            return uH
#        
#        pool = ThreadPool()
#        self.sol = pool.map(process, np.arange(num))
#        pool.close()
#        pool.join()
                
        for i in range(num):
            uH = fe.Function(self.Vu)
            bcD = fe.DirichletBC(self.Vu, self.theta[i], boundaryD)
            left_m, right_m = fe.assemble_system(left, right, bcD)
            fe.solve(left_m, uH.vector(), right_m)
            self.sol.append(uH)
            
    def gene(self, points_m):
        num = len(self.sol)
        ma = np.zeros((len(points_m), num))
        for i in range(num):
            ma[:, i] = [self.sol[i](points) for points in points_m]
        return np.array(ma)

    def geneM(self, funT):
        num = len(self.theta)
        m = funT(np.linspace(0, 1, num), np.ones(num))
        return m
    
    def setBasis(self, m):
        self.mE = m
        
    def basisFun(self, x):
        num = len(self.mE)
        return np.sum([self.mE[i]*self.theta[i](x, 1) for i in range(num)])
    
    def calBasisFun(self, x):
        vbasisFun = np.vectorize(self.basisFun)
        return vbasisFun(x)
        #return np.array([self.basisFun(dian) for dian in x])
        

# used for generate U_H
class GeneUH(object):
    def __init__(self, para):
        self.nx, self.ny = para['mesh_N'][0], para['mesh_N'][1]
        self.mesh = fe.UnitSquareMesh(self.nx, self.ny)
        self.Vu = fe.FunctionSpace(self.mesh, 'P', para['P'])
        self.Vc = fe.FunctionSpace(self.mesh, 'P', para['P'])
        self.al = fe.Expression(para['alpha'], degree=5)
        self.q1 = fe.interpolate(fe.Expression(para['q1'], degree=5), self.Vc)
        self.q2 = fe.interpolate(fe.Expression(para['q2'], degree=5), self.Vc)
        self.q3 = fe.interpolate(fe.Expression(para['q3'], degree=5), self.Vc)
        self.f = fe.Expression(para['f'], degree=5)
        self.theta = fe.Constant(0.0)
        self.u = fe.Function(self.Vu)
        
    def eva(self):
        # construct solutions corresponding to the basis functions
        class BoundaryX0(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 0.0)
               
        class BoundaryX1(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[0], 1.0)
               
        class BoundaryY0(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], 0.0)
               
        class BoundaryY1(fe.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fe.near(x[1], 1.0)
               
        boundaries = fe.MeshFunction('size_t', self.mesh, self.mesh.topology().dim()-1)
        boundaries.set_all(0)
        bc0, bc1, bc2, bc3 = BoundaryX0(), BoundaryX1(), BoundaryY0(), BoundaryY1()
        bc0.mark(boundaries, 1)
        bc1.mark(boundaries, 2)
        bc2.mark(boundaries, 3)
        bc3.mark(boundaries, 4)
        
        domains = fe.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        domains.set_all(0)
        
        dx = fe.Measure('dx', domain=self.mesh, subdomain_data=domains)
        ds = fe.Measure('ds', domain=self.mesh, subdomain_data=boundaries)
        
        u_trial, u_test = fe.TrialFunction(self.Vu), fe.TestFunction(self.Vu)
        left = fe.inner(fe.nabla_grad(u_trial), fe.nabla_grad(u_test))*dx
        right = self.f*u_test*dx + (self.q1*u_test*ds(1) + self.q2*u_test*ds(2) + self.q3*u_test*ds(3))
        bcD = fe.DirichletBC(self.Vu, self.theta, boundaries, 4)
        left_m, right_m = fe.assemble_system(left, right, bcD)
        fe.solve(left_m, self.u.vector(), right_m)
        
    def gene(self, points_m):
        uH = np.array([self.u(points) for points in points_m])
        return uH
  
    
# calculate the full solutions
def calTrueSol(para):
    nx, ny = para['mesh_N'][0], para['mesh_N'][1]
    mesh = fe.UnitSquareMesh(nx, ny)
    Vu = fe.FunctionSpace(mesh, 'P', para['P'])
    Vc = fe.FunctionSpace(mesh, 'P', para['P'])
    al = fe.Constant(para['alpha'])
    f = fe.Constant(para['f'])
    q1 = fe.interpolate(fe.Expression(para['q1'], degree=5), Vc)
    q2 = fe.interpolate(fe.Expression(para['q2'], degree=5), Vc)
    q3 = fe.interpolate(fe.Expression(para['q3'], degree=5), Vc)
    theta = fe.interpolate(fe.Expression(para['q4'], degree=5), Vc)
        
    class BoundaryX0(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], 0.0)
    
    class BoundaryX1(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[0], 1.0)
    
    class BoundaryY0(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[1], 0.0)
    
    class BoundaryY1(fe.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and fe.near(x[1], 1.0)
    
    boundaries = fe.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    boundaries.set_all(0)
    bc0, bc1, bc2, bc3 = BoundaryX0(), BoundaryX1(), BoundaryY0(), BoundaryY1()
    bc0.mark(boundaries, 1)
    bc1.mark(boundaries, 2)
    bc2.mark(boundaries, 3)
    bc3.mark(boundaries, 4)
    
    domains = fe.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains.set_all(0)
    
    bcD = fe.DirichletBC(Vu, theta, boundaries, 4)
    dx = fe.Measure('dx', domain=mesh, subdomain_data=domains)
    ds = fe.Measure('ds', domain=mesh, subdomain_data=boundaries)
    
    u_trial, u_test = fe.TrialFunction(Vu), fe.TestFunction(Vu)
    u = fe.Function(Vu)
    left = fe.inner(al*fe.nabla_grad(u_trial), fe.nabla_grad(u_test))*dx
    right = f*u_test*dx + (q1*u_test*ds(1) + q2*u_test*ds(2) + q3*u_test*ds(3))
    
    left_m, right_m = fe.assemble_system(left, right, bcD)
    fe.solve(left_m, u.vector(), right_m)          
   
    return u
        

def trueFun(x, y):
    return np.sin(np.pi*x)*np.exp(np.pi*y) + x + y


class MeasurePoints(object):
    def __init__(self, num):
        # num is the number of measurment points
        yy = np.linspace(0, 1, np.floor(num/2))
        xxl, xxr = yy*0.0, yy*0.0+1
        self.points_m = np.vstack((np.transpose([xxl, yy]), np.transpose([xxr, yy])))
        self.num = len(self.points_m)
        
        
        
        
        
        
        

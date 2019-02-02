import numpy as np
from scipy.sparse import spdiags

"""
[1] This script used for implementing Alg I, Alg II described in 
B. Jin and J. Zou, Hierarchical Bayesian inference for ill-posed 
problems via variational method, Journal of Computational Physics, 
229, 2010, 7317-7343. 

approxIGaussian is the Alg I in [1]; 
approxIIGaussian is the Alg II in [1]; 
"""

def geneL(num):
    zhu = np.ones(num)
    ci = -1*zhu
    L1 = spdiags([zhu, ci], [0, 1], num-1, num).toarray()
    W = num*(L1.T)@L1 #+ np.eye(num)
    #W = (L1.T)@L1
    return np.array(W)


def findMinL2(H, W, d, eta):
    Ht = np.transpose(H)
    ch1, ch2 = np.shape(H)
    temp = Ht@H + eta*W
    r = Ht@d
    x = np.linalg.solve(temp, r)
    return x


def eignCompu(listM):
    # This function select eigenvalues larger than eps for 
    # all matrixes appeared in listM
    eps = 1e-5
    i = 0
    lenM = len(listM)
    lanMt = []
    lanMtt, lanVecM = np.linalg.eig(listM[0])
    xuan_t = (lanMtt > eps)
    lanMt.append(lanMtt)
    if lenM >= 2:
        for M in listM[1:]:
            lanMtt, lanVecM = np.linalg.eig(M)
            xuan = (lanMtt > eps) & xuan_t
            xuan_t = xuan
            lanMt.append(lanMtt)
    for i in range(lenM):
        lanMt[i] = np.real(lanMt[i][xuan])

    return lanMt


def approxIGaussian(H, W, d, para):
    alpha0, alpha1, beta0, beta1 = para['alpha0'], para['alpha1'], para['beta0'], para['beta1']
    n, m = np.shape(H)
    Ht = np.transpose(H)
    HTH = Ht@H
    alpha02 = m/2 + alpha0
    alpha12 = n/2 + alpha1
    beta0k, beta1k = beta0, beta1
    lan_k = alpha0/beta0
    tau_k = alpha1/beta1
    eta_k = lan_k/tau_k
    eta_km = -10
    #m_k = para['m0']
    
    tol = 1e-3
    ite, max_ite = 1, 1000
    eta_full, lan_full, tau_full = [], [], []
    # -------------------------------------------------------------------------
    # prapare for the evaluation of the trace to avoid small numbers 
    # appeared in the denominator
    lanList = eignCompu([HTH, W])
    lanHTH, lanW = lanList[0], lanList[1]
    # -------------------------------------------------------------------------
    while np.abs(eta_k - eta_km)/np.abs(eta_k) > tol and ite <= max_ite:
        # update q^{k}(m)
        cov_mk = HTH + eta_k*W
        m_k = np.linalg.solve(cov_mk, Ht.dot(d))    
        # update q^{k}(\lambda)
        temp1 = m_k@W@m_k
        temp2 = (np.sum(lanW/(lanHTH + eta_k*lanW)))/tau_k  # needs to take attentation
        beta0k = 0.5*(temp1 + temp2) + beta0
        lan_k = alpha02/beta0k
        # update q^{k}(\tau)
        temp0 = H@m_k - d
        temp1 = np.transpose(temp0)@temp0
        temp2 = (np.sum(lanHTH/(lanHTH + eta_k*lanW)))/tau_k # needs to take attentation
        beta1k = 0.5*(temp1 + temp2) + beta1
        tau_k = alpha12/beta1k
        # update \eta_{k}
        eta_km = eta_k
        eta_k = lan_k/tau_k
        #update 
        ite += 1
        eta_full.append(eta_k)
        lan_full.append(lan_k)
        tau_full.append(tau_k)
        # alternating alpha02 since the \tau (\sigma) converges quickly
        # alpha02 = m/2 + alpha0*np.sqrt(tau_k)
        
    if ite == max_ite:
        print('Maximum iteration number ', max_ite, ' reached')
    
    return m_k, cov_mk, eta_full, lan_full, tau_full, ite


def approxIIGaussian(H, W, d, para):
    alpha0, alpha1, beta0, beta1 = para['alpha0'], para['alpha1'], para['beta0'], para['beta1']
    n, m = np.shape(H)
    Ht = np.transpose(H)
    HTH = Ht@H
    alpha02 = m/2 + alpha0
    alpha12 = n/2 + alpha1
    beta0k, beta1k = beta0, beta1
    lan_k = alpha0/beta0
    tau_k = alpha1/beta1
    eta_k = lan_k/tau_k
    eta_km = -10
    #m_k = para['m0']
    
    tol = 1e-3
    ite, max_ite = 1, 1000
    eta_full, lan_full, tau_full = [], [], []
    while np.abs(eta_k - eta_km)/np.abs(eta_k) > tol and ite <= max_ite:
        # update q^{k}(m)
        cov_mk = HTH + eta_k*W
        m_k = np.linalg.solve(cov_mk, Ht.dot(d))    
        # update q^{k}(\lambda)
        temp1 = m_k@W@m_k
        beta0k = 0.5*temp1+ beta0
        lan_k = alpha02/beta0k
        # update q^{k}(\tau)
        temp0 = H@m_k - d
        temp1 = np.transpose(temp0)@temp0
        beta1k = 0.5*temp1 + beta1
        tau_k = alpha12/beta1k
        # update \eta_{k}
        eta_km = eta_k
        eta_k = lan_k/tau_k
        #update 
        ite += 1
        eta_full.append(eta_k)
        lan_full.append(lan_k)
        tau_full.append(tau_k)
        # alternating alpha02, since the \tau (\sigma) converges quickly
        # alpha02 = m/2 + alpha0*np.sqrt(tau_k)
        
    if ite == max_ite:
        print('Maximum iteration number ', max_ite, ' reached')
    
    return m_k, cov_mk, eta_full, lan_full, tau_full, ite
    

#def approxICenteredT(H, LTL, d, para):
#    alpha0, alpha1 = para['alpha0'], para['alpha1']
#    beta0, beta1 = para['beta0'], para['beta1']
#    n, m = np.shape(H)
#    Ht = np.transpose(H)
#    s = np.linalg.matrix_rank(LTL)
#    alpha0k = alpha0 + s/2.0
#    alpha1k = alpha1 + 1/2.0
#    beta0k, beta1k = beta0, beta1
#    lambda_k = alpha0k/beta0k
#    len_d = len(d)
#    Wk = np.diag(np.repeat(beta0k, len_d))
#    
#    tol = 1e-5
#    ite, max_ite = 1, 1000
#    lan_full, e_full = [], []
#    m_k, m_k1 = [1, 1], [0, 0] 
#    HTH = Ht@H
#    while np.linalg.norm((m_k-m_k1)/m_k, 2) > tol and ite <= max_ite:
#        # update q^{k}(m)
#        HTWH = Ht@Wk@H
#        cov_mk = HTWH + lambda_k*LTL
#        m_k = np.linalg.solve(cov_mk, Ht.dot(Wk.dot(d)))
#        # update q^{k}(w)
#        # prapare for the evaluation of the trace to avoid small numbers 
#        # appeared in the denominator
#        lanList = eignCompu([HTWH, HTH, LTL])
#        lanHTWH, lanHTH, lanLTL = lanList[0], lanList[1], lanList[2]
#        
#        # update q^{k}(\lambda)
#    
#    
#    if ite == max_ite:
#        print('Maximum iteration number ', max_ite, ' reached')
#        
#    return m_k, cov_mk, lan_full, e_full, ite
#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


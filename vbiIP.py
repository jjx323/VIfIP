import numpy as np
from scipy.sparse import spdiags

"""
[1] B. Jin and J. Zou, Hierarchical Bayesian inference for ill-posed 
problems via variational method, Journal of Computational Physics, 
229, 2010, 7317-7343. 
[2] B. Jin, A variational Bayesian method to inverse problems with 
implusive noise, Journal of Computational Physics, 231. 2012, 423-435.

approxIGaussian is the Alg I in [1]; 
approxIIGaussian is the Alg II in [1]; 
approxICenteredT is the Alg I in [2];
"""

def geneL(num):
    zhu = np.ones(num)
    ci = -1*zhu
    L1 = spdiags([zhu, ci], [0, 1], num-1, num).toarray()
    W = num*(L1.T)@L1 #+ np.eye(num)
    W[0, 0] = 2*W[1, 1]   # force the left boundary equal to zero
    W[-1, -1] = 2*W[1, 1] # force the right boundary equal to zero
    #W = (L1.T)@L1
    return np.array(W)


def findMinL2(H, W, d, eta):
    Ht = np.transpose(H)
    ch1, ch2 = np.shape(H)
    temp = Ht@H + eta*W
    r = Ht@d
    x = np.linalg.solve(temp, r)
    return x


def myInv(A, eps=1e-5):
    # This function evaluate the inverse of a matrix by 
    # assigning small singular values to be zero
    U, S, V = np.linalg.svd(A)
    da = S >= eps
    S[da] = 1/S[da]
    S[~da] = 0.0 
    S = np.diag(S)
    return (V.T)@S@(U.T)


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
    while np.abs(eta_k - eta_km)/np.abs(eta_k) > tol and ite <= max_ite:
        # update q^{k}(m)
        precision_mk = HTH + eta_k*W
        #cov_mk = myInv(precision_mk)
        m_k = np.linalg.solve(precision_mk, Ht.dot(d))  
        #m_k = cov_mk@(Ht.dot(d))
        # update q^{k}(\lambda)
        temp1 = m_k@W@m_k
        temp2 = np.trace(np.linalg.solve(precision_mk, W))/tau_k
        #temp2 = np.trace(cov_mk@W)/tau_k
        beta0k = 0.5*(temp1 + temp2) + beta0
        lan_k = alpha02/beta0k
        # update q^{k}(\tau)
        temp0 = H@m_k - d
        temp1 = np.transpose(temp0)@temp0
        temp2 = np.trace(np.linalg.solve(precision_mk, HTH))/tau_k
        #temp2 = np.trace(cov_mk@HTH)/tau_k
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
    
    return m_k, precision_mk, eta_full, lan_full, tau_full, ite


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
        precision_mk = HTH + eta_k*W
        m_k = np.linalg.solve(precision_mk, Ht.dot(d))
        #cov_mk = myInv(precision_mk)
        #m_k = cov_mk@(Ht.dot(d))
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
    
    return m_k, eta_full, lan_full, tau_full, ite
    

def approxICenteredT(H, LTL, d, para):
    alpha0, alpha1 = para['alpha0'], para['alpha1']
    beta0, beta1 = para['beta0'], para['beta1']
    n, m = np.shape(H)
    Ht = np.transpose(H)
    s = np.linalg.matrix_rank(LTL)
    len_d = len(d)
    alpha0k = alpha0 + s/2.0
    alpha1k = alpha1 + 1/2.0
    beta0k = beta0
    beta1k = np.repeat(beta1, len_d)
    lambda_k = alpha0k/beta0k
    Wk = np.diag(alpha1k/beta1k)
    
    tol = 1e-5
    ite, max_ite = 1, 1000
    lan_full, e_full = [], []
    m_k, m_k1 = np.ones(m), np.zeros(m)
    err = np.linalg.norm((m_k-m_k1)/m_k, 2)
    while err > tol and ite <= max_ite:
        m_k1 = m_k.copy()
        # update q^{k}(m)
        precision_mk = Ht@Wk@H + lambda_k*LTL       
        m_k = np.linalg.solve(precision_mk, Ht.dot(Wk.dot(d)))
        #cov_mk = myInv(precision_mk)  
        #m_k = cov_mk@Ht@Wk@d
        # update q^{k}(w)
        temp0 = H@m_k - d
        temp1 = temp0*temp0
        temp2 = np.diag(H@np.linalg.solve(precision_mk, Ht))
        #temp2 = np.diag(H@cov_mk@Ht)
        beta1k = beta1 + 0.5*(temp1 + temp2)
        #beta1k = beta1 + 0.5*(temp1)
        Wk = alpha1k/beta1k
        Wk = np.diag(Wk)   
        # update q^{k}(\lambda)
        precision_mk = Ht@Wk@H + lambda_k*LTL
        temp1 = m_k@LTL@m_k
        temp2 = np.trace(np.linalg.solve(precision_mk, LTL))
        #temp2 = np.trace(cov_mk@LTL)
        beta0k = beta0 + 0.5*(temp1 + temp2)
        #beta0k = beta0 + 0.5*(temp1)
        lambda_k = alpha0k/beta0k
        #update
        err = np.linalg.norm((m_k-m_k1)/m_k, 2)
        ite += 1
        lan_full.append(lambda_k)
        e_full.append(err)
    
    if ite == max_ite:
        print('Maximum iteration number ', max_ite, ' reached')
        
    return m_k, precision_mk, lan_full, e_full[1:], np.diag(Wk), ite
    
    
    
    
#def eignCompu(listM):
#    # This function select eigenvalues larger than eps for 
#    # all matrixes appeared in listM
#    eps = 1e-5
#    i = 0
#    lenM = len(listM)
#    lanMt = []
#    lanMtt, lanVecM = np.linalg.eig(listM[0])
#    xuan_t = (lanMtt > eps)
#    lanMt.append(lanMtt)
#    if lenM >= 2:
#        for M in listM[1:]:
#            lanMtt, lanVecM = np.linalg.eig(M)
#            xuan = (lanMtt > eps) & xuan_t
#            xuan_t = xuan
#            lanMt.append(lanMtt)
#    for i in range(lenM):
#        lanMt[i] = np.real(lanMt[i][xuan])
#
#    return lanMt
    

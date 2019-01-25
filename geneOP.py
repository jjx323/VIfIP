import numpy as np
from scipy.sparse import spdiags

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


def approxI(H, W, d, para):
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
    # ------------------------------------------------------------------------------------------
    # prapare for the evaluation of the trace to avoid small numbers appeared in the denominator
    lanHTHT, lanVec = np.linalg.eig(HTH)
    lanWT, WVec = np.linalg.eig(W)
    eps = 1e-3
    xuan = (lanHTHT > eps) & (lanWT > eps)
    lanHTH = np.real(lanHTHT[xuan])
    lanW = np.real(lanWT[xuan])
    # ------------------------------------------------------------------------------------------
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
        alpha02 = m/2 + alpha0*np.sqrt(tau_k)
        
    if ite == max_ite:
        print('Maximum iteration number ', max_ite, ' reached')
    
    return m_k, cov_mk, eta_full, lan_full, tau_full, ite


def approxII(H, W, d, para):
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
        alpha02 = m/2 + alpha0*np.sqrt(tau_k)
        
    if ite == max_ite:
        print('Maximum iteration number ', max_ite, ' reached')
    
    return m_k, cov_mk, eta_full, lan_full, tau_full, ite
    



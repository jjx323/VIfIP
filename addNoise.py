import numpy as np
import matplotlib.pyplot as plt

def addGaussianNoise(d, para, testMode='n'):   
#    This function add noises as follows:
#    d_i = d_i with probability 1-para['rate']
#    d_i = d_i + \epsilon \xi_i with probability para['rate']
#    where \xi_i follow the standard normal distribution, 
#    para['rate'] is the corruption percentage,
#    para['noise_level'] is the noise level, 
#    
#    The output:
#    d: the noisy data;
#    sig: is the covariance of the added noises.
#    
#    Ref: B. Jin, A variational Bayesian method to inverse problems with implusive noise,
#    Journal of Computational Physics, 231, 2012, 423-435 (Page 428, Section 4). 
    
    if testMode == 'y':
        np.random.seed(1)
    noise_level = para['noise_level']
    len_d = len(d)
    r = para['rate']
    noise = np.random.normal(0, 1, len_d)
    select = (np.random.uniform(0, 1, len_d) < r)
    d = d + noise_level*np.max(np.abs(d))*noise*select
    sig = noise_level*np.max(np.abs(d))
    
    return d, sig

def addImplusiveNoise(d, para, testMode='n'):
#    This function add noise as follows:
#    d_i = d_i with probability 1-para['rate']
#    d_i = d_i + para['noise_levle']*U[-1, 1] with probability para['rate']
#    Here, U[-1, 1] stands for uniform probability distribution between -1 and 1
#    para['rate'] is the corruption percentage,
#    para['noise_level'] is the noise level, 
#    
#    The output:
#    d: the noisy data

    if testMode == 'y':
        np.random.seed(1)
    noise_level = para['noise_level']
    len_d = len(d)
    r = para['rate']
    select = (np.random.uniform(0, 1, len_d) < r)
    noise = np.random.uniform(-1, 1, len_d)
    d = d + noise_level*noise*select
    
    return d

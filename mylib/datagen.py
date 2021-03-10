""" Function that generates artificial data """
import numpy as np

def categorical1(N):
    """ data(samples, features)"""
    
     
    data = np.empty(shape=(N,2))  
    tar = np.empty(shape=(N)) 
    N1 = int(N/2)

    # Positive samples
    data[:N1,:] = np.random.normal(loc=1, scale=0.5, size=(N1,2))
    # Negative samples 
    data[N1:,:] = np.random.normal(loc=-1, scale=0.5, size=(N-N1,2))
    
    
    # Target
    tar[:N1] = np.ones(shape=(N1))
    tar[N1:] = np.zeros(shape=(N-N1))

    return data,tar


def categorical2(N):
    """ data(samples, features)"""

    data = np.empty(shape=(N,2))  
    tar = np.empty(shape=(N)) 
    N1 = int(2*N/3)
    
    # disk
    teta_d = np.random.uniform(0, 2*np.pi, N1) # Random angle
    inner, outer = 2, 5
    r2 = np.sqrt(np.random.uniform(inner**2, outer**2, N1)) # Random radii
    data[:N1,0],data[:N1,1] = r2*np.cos(teta_d), r2*np.sin(teta_d) # Create disk data in pairs (x,y)
        
    #circle
    teta_c = np.random.uniform(0, 2*np.pi, N-N1)
    inner, outer = 0, 3
    r2 = np.sqrt(np.random.uniform(inner**2, outer**2, N-N1))
    data[N1:,0],data[N1:,1] = r2*np.cos(teta_c), r2*np.sin(teta_c)

    tar[:N1] = np.ones(shape=(N1,))
    tar[N1:] = np.zeros(shape=(N-N1,))
    
    return data, tar


def regr1(N, v=0):
    """ data(samples, features)"""

    data = np.empty(shape=(N,6), dtype = np.float32)  
    
    uni = lambda n : np.random.uniform(0,1,n)
    norm = lambda n : np.random.normal(0,1,n)
    noise =  lambda  n : np.random.normal(0,1,n)
    
    
    for i in range(4):
        data[:,i] = norm(N)
    for j in [4,5]:
        data[:,j] = uni(N)
    
    tar =   2*data[:,0] + data[:,1]* data[:,2]**2 + np.exp(data[:,3]) + \
            5*data[:,4]*data[:,5]  + 3*np.sin(2*np.pi*data[:,5])
    std_signal = np.std(tar)
    tar = tar + v * std_signal * noise(N)
        
    return data, tar
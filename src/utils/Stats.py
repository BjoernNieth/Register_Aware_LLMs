import torch
from tqdm import tqdm
import numpy as np
from torch.nn.functional import pdist
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def median_heuristic(X, Y, sample_size=10000):
    if len(X) > sample_size:
        ids = np.random.choice(X.shape[0], sample_size, replace=False)  
        X = X[ids]
    if len(Y) > sample_size:
        ids = np.random.choice(Y.shape[0], sample_size, replace=False)  
        Y = Y[ids]
    Z = np.vstack([X, Y])
    # Compute squared pairwise distances
    G = np.sum(Z**2, axis=1)
    D = G[:, None] + G[None, :] - 2 * np.dot(Z, Z.T)
    
    # Avoid negative due to numerical errors
    D = np.maximum(D, 0)

    # Get upper triangular (excluding diagonal) values
    triu_indices = np.triu_indices_from(D, k=1)
    distances = D[triu_indices]

    # Return sqrt of median squared distance
    return np.sqrt(np.median(distances))
    
def squared_euclidian_XY(X,Y):
    """
    Calculate the squared Euclidian distances between X and Y in a matrix
    """
    
    # Rational of calculation of X and Y (formula include broadcasting of torch):    
    # X(m,d)
    # Y(n,d)
    
    # [X_1^2]    [Y_1^2, Y_2^2, .., Y_n^2]   [X_1^2 + Y_1^2, X_1 + Y_2^2, ..
    # [X_n^2]                                [X_2^2 + Y_1^2, ...
    # [...  ]  +                           = [...
    # [X_m^2]                                
    X_ = X.pow(2).sum(dim=1).unsqueeze(1)
    Y_ = Y.pow(2).sum(dim=1).unsqueeze(0)
    return X_ + Y_ - 2.0 * X@Y.T
    
def MMD2_rbf(X,Y):
    """
    Calculate the empirical unbiased squared MMD between two samples of different size using an RBF kernel.
    The bandwith is set to the median distance of the aggregated samples following the orgiginal "A kernel two sample test" paper.
    """
    # Calculate the bandwith as the median of the distances of the aggregate sample.
    # The bandwith is already squared and doubled to be directly used in the RBF kernel. 
    bandwith = 2*(torch.median(pdist(torch.cat([X,Y])))**2)
    bandwith = max(bandwith, 1e-10) # Avoid division by zero.

    m = X.shape[0]
    n = Y.shape[0]
    # Note: XX and YY distance is calculated using pdist(Upper triangular distance part). This follows from the symmetry of the kernel matrix for the RBF kernel.
    # Instead of claculating full matrix just double result for upper part.
    distance_XX = (2*torch.exp(-torch.square(pdist(X))/bandwith)).sum()/(m*(m-1))
    distance_YY = (2*torch.exp(-torch.square(pdist(Y))/bandwith)).sum()/(n*(n-1))
    distance_XY = (2*torch.exp(-squared_euclidian_XY(X,Y)/bandwith)).sum()/(n*m)
    
    return distance_XX + distance_YY - distance_XY

def mini_batch_MMD(X, Y, batch_size=9500, num_batches=100):
    """ Calculate the empirical unbiased squared MMD between two samples of different size using an RBF kernel.
    Because MMD is in (O^2) we use mini-batches to approximate the MMD following the law of large numbers."""
    mmd = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in tqdm(range(num_batches)):
        if len(X) > batch_size:
            X_batch = X.sample(batch_size)
        else:
            X_batch = X
        if len(Y) > batch_size:
            Y_batch = Y.sample(batch_size)
        else:
            Y_batch = Y
        with torch.no_grad():
            mmd += MMD2_rbf(torch.tensor(X_batch.values).to(device), torch.tensor(Y_batch.values).to(device)).cpu()
            
    return mmd/num_batches

def shuffle_tensor(x):
    return x[torch.randperm(x.size(0))]
        
def MMD_two_sample(s_1, s_2, alpha=0.01, num_perm=1000):

    mmd_s_1_2 = MMD2_rbf(s_1, s_2).cpu().item()
    
    sample_size = s_1.shape[0]
    s_H0 = torch.cat([s_1, s_2])
    
    mmd_perm = []
    for _ in tqdm(range(num_perm)):
        s_H0 = shuffle_tensor(s_H0)
        
        H_resampled = s_H0[:sample_size]
        M_resampled = s_H0[sample_size:]
        mmd_perm.append(MMD2_rbf(H_resampled, M_resampled).cpu().item())
        
    mmd_perm.sort()
    critical_value = mmd_perm[int(len(mmd_perm)* (1-alpha))]
    p_value = np.mean(np.array(mmd_perm) >= mmd_s_1_2)
    is_succesful = critical_value > mmd_s_1_2
    
    return is_succesful, mmd_s_1_2, critical_value, p_value
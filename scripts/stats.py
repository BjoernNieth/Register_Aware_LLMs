import torch
from tqdm import tqdm
from torch.nn.functional import pdist
import numpy as np
from scipy.stats import wasserstein_distance

def permutation_test_wasserstein_full(x, y, n_permutations=1000, seed=42):
    rng = np.random.default_rng(seed)
    
    W_obs = wasserstein_distance(x, y)
    z = np.concatenate([x, y])
    n_x = len(x)
    
    perm_stats = np.zeros(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(z)
        perm_stats[i] = wasserstein_distance(z[:n_x], z[n_x:])
    
    p_value = np.mean(perm_stats >= W_obs)
    #ci_low, ci_high = np.percentile(perm_stats, [2.5, 97.5])
    
    return W_obs, p_value#, ci_low, ci_high

def wasserstein_bootstrap(dim_human,
                            dim_model,
                             n=600,
                             R=1000,
                             ci=95,
                             seed=42):
    """ Performe a bootstrap between the human and model distribution """
    rng = np.random.default_rng(seed)
    N = dim_human.shape[0]
    num_dims = dim_human.shape[1]
    wd_scores = [[]] * num_dims
    
    for _ in range(R):
        # Draw the ids for the bootstrap
        idx= rng.choice(N, size=n, replace=True)

        # Draw a the human and model sample
        A = dim_human[idx]
        B = dim_model[idx]

        for d in range(num_dims):
            wd_scores[d].append(wasserstein_distance(A[:, d], B[:, d]))
    
    means = []
    stds = []
    lower_cis = []
    upper_cis = []
    observed_wds = []
    for d in range(num_dims):
        observed_wds.append(wasserstein_distance(dim_human[:, d], dim_model[:, d]))
        wd_scores_d = np.array(wd_scores[d])
        means.append(wd_scores_d.mean())
        stds.append(wd_scores_d.std())
        lower_cis.append(np.percentile(wd_scores_d,  0 + (100-ci)/2))
        upper_cis.append(np.percentile(wd_scores_d, 100 - (100-ci)/2))
    return observed_wds, means, stds, lower_cis, upper_cis


def mmd_permutation_test(X, Y, R=1000, sigma=None):
    n = X.shape[0]
    # observed statistic
    obs_mmd2, sigma = compute_mmd2_biased_batched(X, Y, sigma)
    obs_mmd2 = obs_mmd2.item()
    # build null distribution
    Z = np.vstack([X, Y])  # combined (1200, D)
    N = Z.shape[0]
    null_stats = []

    rng = np.random.default_rng(127)

    for _ in range(R):
        perm = rng.permutation(N)
        A = Z[perm[:n]]
        B = Z[perm[n:]]
        mmd2_null, _ = compute_mmd2_biased_batched(A, B, sigma=sigma)  
        null_stats.append(mmd2_null)

    null_stats = np.array(null_stats)
    p_value = (np.sum(null_stats >= obs_mmd2)) / R

    return obs_mmd2, p_value

def mmd_bootstrap(dim_human,
                  dim_model,
                             n=600,
                             R=1000,
                             sigma=None,
                             ci=95,
                             seed=42):
    """ Performe a bootstrap between the human and model distribution """
    rng = np.random.default_rng(seed)
    N = dim_human.shape[0]
    hm_scores = []
    
    for _ in range(R):
        # Draw the ids for the bootstrap
        idx= rng.choice(N, size=n, replace=True)

        # Draw a the human and model sample
        A = dim_human[idx]
        B = dim_model[idx]

        mmd2_hm, _ = compute_mmd2_biased_batched(A, B, sigma=sigma)
        hm_scores.append(mmd2_hm)

    hm_scores = np.array(hm_scores)
    hm_mean = hm_scores.mean()
    hm_std = hm_scores.std()
    lower = np.percentile(hm_scores,  0 + (100-ci)/2)
    upper = np.percentile(hm_scores, 100 - (100-ci)/2)
    return hm_scores, hm_mean, hm_std, lower, upper

def human_human_variance_baseline(feature_human,
                                n=600,
                                R=1000,
                                ci=0.95,
                                seed=42):
    rng = np.random.default_rng(seed)
    N = feature_human.shape[0]
    hh_variances = []
    
    for _ in range(R):
        perm = rng.permutation(N)
        idx_A = perm[:n]

        A = feature_human[idx_A]

        var_A = np.var(A, axis=0, ddof=1).sum()
        hh_variances.append(var_A)
        
    hh_variances = np.array(hh_variances)
    hh_mean = hh_variances.mean()
    hh_std = hh_variances.std()
    lower = np.percentile(hh_variances,  0 + (100-ci)/2)
    upper = np.percentile(hh_variances, 100 - (100-ci)/2)
    return hh_variances, hh_mean, hh_std, lower, upper



def human_human_mmd_baseline(feature_human,
                             n=600,
                             R=1000,
                             sigma=None,
                             ci=0.95,
                             seed=42,
                             centering=False):
    rng = np.random.default_rng(seed)
    N = feature_human.shape[0]
    hh_scores = []
    
    for _ in range(R):
        perm = rng.permutation(N)
        idx_A = perm[:n]
        idx_B = perm[n:2*n]

        A = feature_human[idx_A]
        B = feature_human[idx_B]

        # Remove the mean from both samples to focus on the spread rather than the location
        if centering:
            A = A - A.mean(axis=0, keepdims=True)
            B = B - B.mean(axis=0, keepdims=True)

        mmd2_hh, _ = compute_mmd2_biased_batched(A, B, sigma=sigma)
        hh_scores.append(mmd2_hh)

    hh_scores = np.array(hh_scores)
    hh_mean = hh_scores.mean()
    hh_std = hh_scores.std()
    lower = np.percentile(hh_scores,  0 + (100-ci)/2)
    upper = np.percentile(hh_scores, 100 - (100-ci)/2)
    return hh_scores, hh_mean, hh_std, lower, upper


def human_human_wasserstein_baseline(dim_human,
                             n=600,
                             R=1000,
                             ci=0.95,
                             seed=42):
    rng = np.random.default_rng(seed)
    N = dim_human.shape[0]
    n_dims = dim_human.shape[1]
    baseline_dists = [[] for _ in range(n_dims)]
    for _ in range(R):
        perm = rng.permutation(N)
        idx_A = perm[:n]
        idx_B = perm[n:2*n]

        A = dim_human[idx_A]
        B = dim_human[idx_B]
        for d in range(n_dims):
            wd = wasserstein_distance(A[:, d], B[:, d])
            baseline_dists[d].append(wd)
        
    cis_lower = []
    cis_upper = []
    means = []
    stds = []
    for d in range(n_dims):
        wd_d = np.array(baseline_dists[d])
        means.append(wd_d.mean())
        stds.append(wd_d.std())
        cis_lower.append(np.percentile(wd_d,  0 + (100-ci)/2))
        cis_upper.append(np.percentile(wd_d, 100 - (100-ci)/2))
    return means, stds, cis_lower, cis_upper


def rbf_kernel(x, y, sigma):
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-dist / (2 * sigma ** 2))
    
def estimate_median_heuristic(Z, max_samples=9500):
    # Cast to torch if 
    if type(Z) == np.ndarray:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            Z = torch.tensor(Z).to(device)
    n = Z.shape[0]
    if n > max_samples:
        idx = torch.randperm(n, device=Z.device)[:max_samples]
        Z = Z[idx]

    dot = Z @ Z.t()
    sq_norms = torch.diag(dot)
    D2 = sq_norms.unsqueeze(1) - 2 * dot + sq_norms.unsqueeze(0)

    # Keep only positive off-diagonal entries
    D2 = D2[D2 > 0]

    # Median of distances
    median_val = torch.median(D2)
    sigma = torch.sqrt(0.5 * median_val)

    return sigma
    
def compute_mmd_unbiased_batched(X, Y, sigma=None,batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X).to(device)
    Y = torch.tensor(Y).to(device)
    XY = torch.cat([X, Y], dim=0)
    #mean = XY.mean(dim=0, keepdim=True)
    #std = XY.std(dim=0, keepdim=True) + 1e-8   # numerical stability

    #X = (X - mean) / std
    #Y = (Y - mean) / std
    n, m = X.size(0), Y.size(0)
    if not sigma:
        sigma = estimate_median_heuristic(torch.cat([X, Y]))
    # Compute XX
    xx_sum = 0.0
    for i in range(0, n, batch_size):
        x1 = X[i:i+batch_size]
        for j in range(0, n, batch_size):
            x2 = X[j:j+batch_size]
            k_xx = rbf_kernel(x1, x2, sigma)
            if i == j:
                k_xx = k_xx - torch.diag(torch.diagonal(k_xx))  # remove diagonal
            xx_sum += k_xx.sum().cpu()
    mmd_xx = xx_sum / (n * (n - 1))

    # Compute YY
    yy_sum = 0.0
    for i in range(0, m, batch_size):
        y1 = Y[i:i+batch_size]
        for j in range(0, m, batch_size):
            y2 = Y[j:j+batch_size]
            k_yy = rbf_kernel(y1, y2, sigma)
            if i == j:
                k_yy = k_yy - torch.diag(torch.diagonal(k_yy))
            yy_sum += k_yy.sum().cpu()
    mmd_yy = yy_sum / (m * (m - 1))

    # Compute XY
    xy_sum = 0.0
    for i in range(0, n, batch_size):
        x1 = X[i:i+batch_size]
        for j in range(0, m, batch_size):
            y1 = Y[j:j+batch_size]
            k_xy = rbf_kernel(x1, y1, sigma)
            xy_sum += k_xy.sum().cpu()
    mmd_xy = xy_sum * 2 / (n * m)

    mmd2 = mmd_xx + mmd_yy - mmd_xy
    return mmd2, sigma


def mk_mmd_biased_batched(X, Y, sigma=None, kernel_scales=None):
    if kernel_scales is None:
        kernel_scales = [pow(2, i/2) for i in range(-4, 4 + 1)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sigma == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sigma = estimate_median_heuristic(torch.cat([torch.tensor(X).to(device), torch.tensor(Y).to(device)]))

    mmd_2 = 0
    for kernel_scale in kernel_scales:
        mmd_2 += compute_mmd2_biased_batched(X, Y, sigma=sigma*kernel_scale)[0].cpu()

    return mmd_2/len(kernel_scales)

def compute_mmd2_biased_batched(X, Y, sigma=None, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(X, device=device, dtype=torch.float32)
    Y = torch.as_tensor(Y, device=device, dtype=torch.float32)
    n = X.size(0)
    m = Y.size(0)

    if sigma is None:
        sigma = estimate_median_heuristic(torch.cat([X, Y], dim=0))

    # XX: include diagonal, divide by n^2
    xx_sum = 0.0
    for i in range(0, n, batch_size):
        x1 = X[i:i+batch_size]
        for j in range(0, n, batch_size):
            x2 = X[j:j+batch_size]
            xx_sum += rbf_kernel(x1, x2, sigma).sum()
    mmd_xx = xx_sum / (n * n)

    # YY: include diagonal, divide by m^2
    yy_sum = 0.0
    for i in range(0, m, batch_size):
        y1 = Y[i:i+batch_size]
        for j in range(0, m, batch_size):
            y2 = Y[j:j+batch_size]
            yy_sum += rbf_kernel(y1, y2, sigma).sum()
    mmd_yy = yy_sum / (m * m)

    # XY: divide by nm and multiply by 2
    xy_sum = 0.0
    for i in range(0, n, batch_size):
        x1 = X[i:i+batch_size]
        for j in range(0, m, batch_size):
            y1 = Y[j:j+batch_size]
            xy_sum += rbf_kernel(x1, y1, sigma).sum()
    mmd_xy = (2.0 * xy_sum) / (n * m)

    mmd2 = mmd_xx + mmd_yy - mmd_xy
    return mmd2.detach().cpu(), sigma


def get_cohens_d(mean_1, mean_2, std_1, std_2, d_1, d_2):
    std_pooled = np.sqrt(((d_1-1)*std_1 + (d_2-1)*std_2)/(d_1+d_2-2))
    return (mean_1 - mean_2) / std_pooled

if __name__ == "__main__":  
    print("Init Z")
    Z = np.random.normal(1, 1, (600, 1))
    S_1 = np.concatenate([Z, -Z, Z, -Z, Z], axis=1)
    S_2 = np.concatenate([-Z, Z, -Z, Z, -Z], axis=1)
    print(S_1.shape)
    print("Z = Normal(0,1)")
    print(f"MK-MMD^2([Z, -Z, Z, -Z, Z], [Z, -Z, Z, -Z, Z]): {mk_mmd_biased_batched(S_1, S_1)}")
    print(f"MK-MMD^2([Z, -Z, Z, -Z, Z], [-Z, Z, -Z, Z, -Z]): {mk_mmd_biased_batched(S_1, S_2)} ")
    print(f"MMD^2([Z, -Z, Z, -Z, Z], [Z, -Z, Z, -Z, Z]): {compute_mmd2_biased_batched(S_1, S_1)[0]}")
    print(f"MMD^2([Z, -Z, Z, -Z, Z], [-Z, Z, -Z, Z, -Z]): {compute_mmd2_biased_batched(S_1, S_2)[0]} ")
    print(f"Mean, Std [Z, Z]: {(S_1.mean(axis=0), S_1.std(axis=0))} ")
    print(f"Mean, Std [Z, -Z]: {(S_2.mean(axis=0), S_2.std(axis=0))} ")


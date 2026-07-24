'''
Benchmark Methods for comparisons with the useful util functions.
'''
def online_fdr_lord(pvals, alpha=0.05):
    n = len(pvals)
    pvals = np.asarray(pvals, float)
    gamma = _gamma_seq(n)
    alpha_t = np.zeros(n)
    reject = np.zeros(n, dtype=bool)
    rej_times = []
    W = alpha
    for t in range(n):
        if np.isnan(pvals[t]) or W <= 1e-12:
            continue
        a = gamma[t] * alpha
        for tau in rej_times:
            lag = t - tau
            if 0 < lag < n:
                a += gamma[lag] * alpha
        alpha_t[t] = min(a, W)
        reject[t] = pvals[t] <= alpha_t[t]
        W = W - alpha_t[t] + (alpha if reject[t] else 0.0)
        if reject[t]:
            rej_times.append(t)
    return alpha_t, reject

def online_fdr_lond(pvals, alpha=0.05):
    n = len(pvals)
    pvals = np.asarray(pvals, float)
    gamma = _gamma_seq(n)
    alpha_t = np.zeros(n)
    reject = np.zeros(n, dtype=bool)
    nr = 0
    for t in range(n):
        if np.isnan(pvals[t]):
            continue
        alpha_t[t] = alpha * gamma[t] * (1 + nr)
        reject[t] = pvals[t] <= alpha_t[t]
        if reject[t]:
            nr += 1
    return alpha_t, reject

def exponential_decay_empirical_pvalue(T_history, T_new, lam = 0.95):
    t = len(T_history)
    if t == 0:
        return 1.0
    T_history = np.asarray(T_history, dtype = float)
    arr = np.arange(t)
    weights = np.exp(- lam * (t -1 - arr))
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return 1.0
    weighted_count = np.dot((T_history >= T_new), weights)
    return (1.0 + weighted_count)/(1.0 + weight_sum)

def empirical_pval_exponential_decay(stats, lam = 0.95):
    n = len(stats)
    pvals = np.ones(n)
    for t in range(1, n):
        pvals[t] = exponential_decay_empirical_pvalue(stats[:t], stats[t], lam = lam)
    return pvals

'''
Martingale Base Methods
'''
def method_martingale2(pvals, alpha = 0.05, eps = 1e-8, lam = 4):
    pvals = np.asarray(pvals, dtype = float)
    pvals = np.clip(pvals, eps, 1.0)
    evals = lam / (np.exp(lam) - 1) * np.exp(lam * (1 - pvals))
    return np.cumprod(evals) > (1.0 / alpha)

def method_martingale3(pvals, alpha=0.05, eps=1e-10):
    """Power martingale: e_t = (2/3)*p_t^(-1/3). E[e]=1 for uniform p. M_t = prod e_i."""
    pvals = np.asarray(pvals, dtype=float)
    pvals = np.clip(pvals, eps, 1.0)
    evals = (2.0 / 3.0) * np.power(pvals, -1.0 / 3.0)
    M = np.cumprod(evals)
    return M > (1.0 / alpha)

def method_martingale4(pvals, alpha=0.05, eps=1e-10):
    """Power martingale: e_t = (4/5)*p_t^(-0.2). E[e]=1 for uniform p. M_t = prod e_i."""
    pvals = np.asarray(pvals, dtype=float)
    pvals = np.clip(pvals, eps, 1.0)
    evals = (4.0 / 5.0) * np.power(pvals, -0.2)
    M = np.cumprod(evals)
    return M > (1.0 / alpha)

def method_martingale(pvals, alpha=0.05, eps=1e-10):
    pvals = np.asarray(pvals, dtype=float)
    pvals = np.clip(pvals, eps, 1.0)
    evals = 1.0 / (2.0 * np.sqrt(pvals))
    M = np.ones(len(pvals))
    M[0] = evals[0]
    for t in range(1, len(pvals)):
        M[t] = M[t - 1] * evals[t]
    return M > (1.0 / alpha)

def method_evalue(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    e_values = 1.0 / np.maximum(pvals, 1e-12)
    wealth = 1.0
    rej = np.zeros(len(pvals), dtype=bool)
    threshold = 1.0 / alpha
    for i, e in enumerate(e_values):
        wealth *= e
        if wealth >= threshold:
            rej[i] = True
            wealth = 1.0
    return rej
def empirical_pval(MSE_dif_list, burnin = 5):
    n = len(MSE_dif_list)
    pval_list = np.zeros(n)
    pval_list[:burnin] = 1
    for i in range(1, n):
        pval_list[i] = (1.0 + np.sum(np.asarray(MSE_dif_list[burnin:(i-1)]) > MSE_dif_list[i]))/i
    return pval_list

def method_fix_alpha(pvals, alpha = 0.05):
    pvals = np.asarray(pvals, float)
    return pvals <= alpha


def method_saffron(pvals, alpha = 0.05, lam = 0.3, tau = 0.5, s = 1.0):
    n = len(pvals)
    pvals = np.asarray(pvals, float)
    g = (1.0/np.arange(1,n+1))/np.sum((1.0/np.arange(1,n+1)))
    nc = 0
    W = alpha
    rej = np.zeros(n, dtype = bool)
    for i in range(n):
        if np.isnan(pvals[i]):
            continue
        if pvals[i] <= lam or pvals[i] > tau:
            W += tau
            continue
        nc += 1
        a = min((tau - lam) * g[nc - 1] * W, W)
        rej[i] = pvals[i] <= a
        W = W - a + (alpha if rej[i] else 0.0)
    return rej

def method_addis(pvals, alpha = 0.05, lam = 0.001, tau = 0.5, s = 1.0):
    n = len(pvals)
    pvals = np.asarray(pvals, float)
    g = (1.0/np.arange(1,n+1))/np.sum((1.0/np.arange(1,n+1)))
    W = alpha 
    nc = 0
    rej = np.zeros(n, dtype = bool)
    for i in range(n):
        if np.isnan(pvals[i]):
            continue
        if pvals[i] <= lam or pvals[i] > tau:
            W += tau
            continue
        nc += 1
        a = min((tau - lam) * g[nc - 1] * W, W)
        rej[i] = pvals[i] <= a
        W = W - a + (alpha if rej[i] else 0.0)
    return rej

def first_k_consecutive_rej(rej, k):
    n = len(rej)
    for i in range(n - k):
        if np.all(rej[i:(i+k)]):
            return (i+k)
    return np.nan

def first_k_consecutive_rej_ind(rej, k):
    n = len(rej)
    for i in range(n - k):
        if rej[(i+k)] - rej[i] == k:
            return rej[i+k]
    return np.nan

class PageHinkley:
    def __init__(self, delta=0.005, threshold=50.0, min_instances=30):
        self.delta, self.threshold, self.min_instances = delta, threshold, min_instances
        self.n, self.x_sum, self.PH_n, self.PH_min = 0, 0.0, 0.0, 0.0
    def update(self, x):
        self.n += 1
        self.x_sum += x
        m_n = self.x_sum / self.n
        self.PH_n += (x - m_n - self.delta)
        self.PH_min = min(self.PH_min, self.PH_n)
        if self.n < self.min_instances:
            return False
        return (self.PH_n - self.PH_min) > self.threshold


import numpy as np
import torch
from torch.nn import Conv1d
from scipy.stats import ranksums, mood, mannwhitneyu, ks_2samp, cramervonmises_2samp
class EWMA():
    """Exponentially Weighted Moving Average algorithm.
    Read more in 
    Parameters
    ----------
    r : float, default=0.1
        Control parameter of the EWMA algorithm monitoring the learning rate
        of the exponentially moving weighted average Z (between 0 and 1).
    L : float, default=2.4.
        Control parameter of the EWMA algorithm used as a threshold for the 
        decision rule and controlling the bandwith.
    burnin : int, default=50
        Number of firts observed values processed before a changepoint can be
        detected. 
    mu : float, default=0.
        Initial mean value of the stream. Recall that CUSUM assumes that 
        observations are normally distributed.
    sigma : float, default=1.
        Initial standard deviation of the stream.
    Attributes
    ----------
    _mu : ndarray of shape (n_samples,)
        Recording of all mean values.
    _sigma : ndarray of shape (n_samples,)
        Recording of all standard deviation values.
    Z : ndarray of shape (n_samples,)
        Z statistic calculated sequentially after processing each new observation, 
        smoothing the original data stream.
    sigma_Z : ndarray of shape (n_samples,)
        Standard deviation of the Z statistic.
    changepoints : ndarray
        Array containing detected changepoints, initialised empty.
    n : int
        Number of observations in the current run length.
    """
    def __init__(self, 
                 r: float = 0.1, 
                 L: float = 2.4, 
                 burnin: int = 50, 
                 mu: float = 0., 
                 sigma:float = 1.):
        self.r = r
        self.L = L
        self.burnin = burnin
        self.mu = mu
        self.sigma = sigma
        self._mu = [mu]
        self._sigma = [sigma]
        self.Z = [mu]
        self.sigma_Z = [0.]
        self.changepoints = []
        self.n = 2
    def update_mean_variance(self, 
                             data_new: float):
        """Update efficiently mean and variance.
        Parameters
        data_new : float
            New observation in the data stream. The mean and variance are updated
            efficiently and online without storing every observed values. 
        """
        mu_new = self.mu + (data_new - self.mu) / self.n
        self.sigma = (self.sigma ** 2 + ((data_new - self.mu) * (data_new - mu_new) - self.sigma ** 2) / self.n) ** 0.5
        self.mu = mu_new
        self._mu.append(mu_new)
        self._sigma.append(self.sigma)
    def update_statistics(self, 
                          i: int, 
                          data_new: float):
        """Update the algorithm statistics Z and sigma_Z.

        Parameters
        ----------
        
        i : int
            Time index.
        
        data_new : float
            New observation in the data stream. Z and sigma_Z are updated according to
            EWMA algorithm formulas.
        """
        self.Z.append((1 - self.r) * self.Z[-1] + self.r * data_new)
        self.sigma_Z.append(self.sigma * ((self.r / (2 - self.r)) * (1 - (1 - self.r) ** (2 * i))) ** 0.5)
    def decision_rule(self, 
                      i: int):
        """Decide whether or not a change has occurred.

        Parameters
        ----------
            
        i : int
            Time index. The decision rule |Z - mu| / sigma_Z > L is implemented in this method.
        """
        if (i >= self.burnin) and (abs((self.Z[-1] - self.mu) / self.L) > self.sigma_Z[-1]):
            self.changepoints.append(i)
            self.n = 2
        else:
            self.n += 1
    def process(self, 
                data: list):
        """Run EWMA algorithm on a univariate data stream.

        Parameters
        ----------
            
        data : list
            Univariate data stream to be processed. The method sequentially first updates mean 
            and variance, then updates the Z and sigma_Z statistics and finally applies the simple
            decision rule to assert if a change has occurred.
        """
        for i in range(1, len(data)):
            self.update_mean_variance(data[i])
            self.update_statistics(i, data[i])
            self.decision_rule(i)




class TwoSample():
    """Two sample test for changepoint detection.
    
    Read more in
    
    Parameters
    ----------
    
    statistic : str, default="Lepage"
        Test statistic to be used by the algorithm. Use 'Mann-Whitney' for changes
        in the location, 'Mood' for changes in the scale, 'Lepage' for changes in 
        both location and scale, 'Kolmogorov-Smirnov' and 'Cramer-von-Mises' for 
        general changes in distribution.
    
    threshold : float, default=3.1
        Threshold value for the test statistic, must be suited for each statistic.
        
    Attributes
    ----------
    
    t : int
        Time index.
    
    changepoints : list
        Array containing detected changepoints, initialised empty. 
    """
    def __init__(self, 
                 statistic: str = "Lepage", 
                 threshold: float = 3.1):
        self.threshold = threshold
        self.statistic = statistic
        self.D = [1.]
        self.t = 2
        self.changepoints = []
    def fetch_statistic(self):
        # Maps statistic attribute with SciPy method
        db = {
            "Mann-Whitney": mannwhitneyu, 
            "Mood": mood, 
            "Lepage": ranksums, 
            "Kolmogorov-Smirnov": ks_2samp,
            "Cramer-von-Mises": cramervonmises_2samp
        }
        self.statistic = db[self.statistic]
    def process_batch(self, 
                      X: list):
        """Process a data stream until a change is detected.

        Parameters
        ----------
            
        X : list
            Univariate array to be processed.

        Returns
        -------
        
        tau : int or None
            First detected changepoint in X. If no change is detected, the method
            returns None.
        """
        Dn = []
        tau = 0
        self.t = 2
        while self.t < len(X):
            Dkn = []
            for k in range(1, self.t):
                x, y = X[tau:k], X[k:self.t]
                # Different syntax for each Scipy test
                try:
                    Dkn.append(abs(self.statistic(x, y)[0]))
                except:
                    try:
                        Dkn.append(abs(self.statistic(x, y).statistic))
                    except: # Edge cases with sample sizes 
                        Dkn.append(0)
            self.D.append(max(Dkn))
            if max(Dkn) < self.threshold:
                Dn.append(max(Dkn))
                self.t += 1
            else:
                tau = self.t
                return tau
        return None
    def process(self, 
                data: list):
        """Run the two-sample test algorithm.
        Parameters
        ----------
        data : list
            Univariate data stream to be processed. The method first tries to detect
            a change in the data. If a change is detected, it will look for changes in the
            remaining sequence, etc. until no change is detected. When no change is detected
            and the whole sequence has been processed, the method stops.
        """
        self.fetch_statistic()
        cp = []
        tau = 0
        while tau is not None:
            data = data[tau:]
            tau = self.process_batch(data)
            self.D.append(self.D[-1])
            cp.append(tau)
        self.changepoints = np.cumsum(cp[:-1])

def empirical_pval(MSE_dif_list, burnin = 5):
    n = len(MSE_dif_list)
    pval_list = np.zeros(n)
    pval_list[:burnin] = 1
    for i in range(1, n):
        pval_list[i] = (1.0 + np.sum(np.asarray(MSE_dif_list[burnin:(i-1)]) > MSE_dif_list[i]))/i
    return pval_list

def method_fix_alpha(pvals, alpha = 0.05):
    pvals = np.asarray(pvals, float)
    return pvals <= alpha


def method_saffron(pvals, alpha = 0.05, lam = 0.3, tau = 0.5, s = 1.0):
    n = len(pvals)
    pvals = np.asarray(pvals, float)
    g = (1.0/np.arange(1,n+1))/np.sum((1.0/np.arange(1,n+1)))
    nc = 0
    W = alpha
    rej = np.zeros(n, dtype = bool)
    for i in range(n):
        if np.isnan(pvals[i]):
            continue
        if pvals[i] <= lam or pvals[i] > tau:
            W += tau
            continue
        nc += 1
        a = min((tau - lam) * g[nc - 1] * W, W)
        rej[i] = pvals[i] <= a
        W = W - a + (alpha if rej[i] else 0.0)
    return rej

def method_addis(pvals, alpha = 0.05, lam = 0.001, tau = 0.5, s = 1.0):
    n = len(pvals)
    pvals = np.asarray(pvals, float)
    g = (1.0/np.arange(1,n+1))/np.sum((1.0/np.arange(1,n+1)))
    W = alpha 
    nc = 0
    rej = np.zeros(n, dtype = bool)
    for i in range(n):
        if np.isnan(pvals[i]):
            continue
        if pvals[i] <= lam or pvals[i] > tau:
            W += tau
            continue
        nc += 1
        a = min((tau - lam) * g[nc - 1] * W, W)
        rej[i] = pvals[i] <= a
        W = W - a + (alpha if rej[i] else 0.0)
    return rej

def first_k_consecutive_rej(rej, k):
    n = len(rej)
    for i in range(n - k):
        if np.all(rej[i:(i+k)]):
            return (i+k)
    return np.nan

def first_k_consecutive_rej_ind(rej, k):
    n = len(rej)
    for i in range(n - k):
        if rej[(i+k)] - rej[i] == k:
            return rej[i+k]
    return np.nan

class PageHinkley:
    def __init__(self, delta=0.005, threshold=50.0, min_instances=30):
        self.delta, self.threshold, self.min_instances = delta, threshold, min_instances
        self.n, self.x_sum, self.PH_n, self.PH_min = 0, 0.0, 0.0, 0.0
    def update(self, x):
        self.n += 1
        self.x_sum += x
        m_n = self.x_sum / self.n
        self.PH_n += (x - m_n - self.delta)
        self.PH_min = min(self.PH_min, self.PH_n)
        if self.n < self.min_instances:
            return False
        return (self.PH_n - self.PH_min) > self.threshold

        

#ocpdet utils:
import numpy as np
import torch
from torch.nn.optim import Adam
from torch.nn import Conv1d
from scipy.stats import ranksums, mood, mannwhitneyu, ks_2samp, cramervonmises_2samp
class EWMA():
    """Exponentially Weighted Moving Average algorithm.
        
    Read more in 
    payment_clean = (payment.sort_values('payment_time').
    groupby('order_id').last().reset_index())
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



class NeuralNetwork():
    def __init__(self, k = 10, n = 5, lag = 100,
        f = None, r = 0.1, L = 3.0, burnin = 100,
        method = 'bump', timeout = 100):
        self.k = k
        self.n = n
        self.l = lag
        if f is None:
            self.f = nn.Sequential([
            	])































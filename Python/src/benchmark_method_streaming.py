import numpy as np
from scipy.stats import norm
from river import drift
from river.anomaly import HalfSpaceTrees
from river import stats
from river import compose

class ExponentialWeightedPValue:
    def __init__(self, lam=0.95, burnin=5):
        self.lam = lam
        self.burnin = burnin
        self.history = []
        self.cum_weight = 0.0
        self.weighted_count = 0.0 
        self.total_weight = 0.0
        self.n = 0
        
    def update(self, new_val):
        self.n += 1
        if self.n <= self.burnin:
            self.history.append(new_val)
            return 1.0
        if self.n > self.burnin + 1:
            t = self.n - 1 
            window = self.history[-2000:] if len(self.history) > 2000 else self.history
            t_w = len(window)
            weights = np.exp(-self.lam * (t_w - 1 - np.arange(t_w)))
            weight_sum = weights.sum()
            if weight_sum == 0:
                pval = 1.0
            else:
                weighted_greater = np.dot((np.array(window) > new_val).astype(float), weights)
                pval = (1.0 + weighted_greater) / (1.0 + weight_sum)
        else:
            pval = 1.0
        
        self.history.append(new_val)
        if len(self.history) > 5000:
            self.history = self.history[-5000:]
        return pval

class OnlineFDR:
    def __init__(self, method='LORD', alpha=0.05, gamma_seq=None, lam=0.3, tau=0.5):
        self.method = method
        self.alpha = alpha
        self.lam = lam
        self.tau = tau
        self.gamma_seq = gamma_seq if gamma_seq is not None else self._gamma_seq
        self.n = 0
        self.rej_times = []   
        self.W = alpha       
        self.nr = 0          
        self.nc = 0         
        self.alpha_t_hist = []
        self.reject_hist = []
    
    def _gamma_seq(self, n):
        pass
    def update(self, pval):
        self.n += 1
        if np.isnan(pval) or pval <= 0:
            pval = 1e-12
        if self.method == 'LORD':
            gamma = 1.0 / (self.n + 1) / (np.log(self.n + 1) + 0.577)  # 近似
            # 计算 sum_{tau in rej_times} gamma_{t - tau}
            sum_gamma = 0.0
            for tau in self.rej_times:
                lag = self.n - 1 - tau
                if lag > 0:
                    sum_gamma += 1.0 / (lag + 1) / (np.log(lag + 1) + 0.577)
            alpha_t = min(self.alpha * (gamma + sum_gamma), self.W)
            reject = pval <= alpha_t
            if reject:
                self.rej_times.append(self.n - 1)
                self.W = self.W - alpha_t + self.alpha
            else:
                self.W = self.W - alpha_t
        elif self.method == 'LOND':
            gamma = 1.0 / (self.n + 1) / (np.log(self.n + 1) + 0.577)
            alpha_t = self.alpha * gamma * (1 + self.nr)
            reject = pval <= alpha_t
            if reject:
                self.nr += 1
        elif self.method == 'SAFFRON':
            if pval <= self.lam or pval > self.tau:
                self.W += self.tau
                reject = False
            else:
                self.nc += 1
                gamma = 1.0 / (self.nc + 1) / (np.log(self.nc + 1) + 0.577)
                alpha_t = min((self.tau - self.lam) * gamma * self.W, self.W)
                reject = pval <= alpha_t
                self.W = self.W - alpha_t + (self.alpha if reject else 0.0)
        elif self.method == 'ADDIS':
            if pval <= self.lam or pval > self.tau:
                self.W += self.tau
                reject = False
            else:
                self.nc += 1
                gamma = 1.0 / (self.nc + 1) / (np.log(self.nc + 1) + 0.577)
                alpha_t = min((self.tau - self.lam) * gamma * self.W, self.W)
                reject = pval <= alpha_t
                self.W = self.W - alpha_t + (self.alpha if reject else 0.0)
        else:
            raise ValueError("Unsupported method")
        
        self.alpha_t_hist.append(alpha_t if 'alpha_t' in locals() else 0)
        self.reject_hist.append(reject)
        return reject

class StreamingMartingale:
    def __init__(self, alpha=0.05, power=0.5, eps=1e-10):
        self.alpha = alpha
        self.power = power  # 0.5 => sqrt, 1/3 => power 3, etc.
        self.eps = eps
        self.M = 1.0
        self.reject = False
    
    def update(self, pval):
        pval = max(pval, self.eps)
        e = 1.0 / (2.0 * np.sqrt(pval))
        self.M *= e
        if self.M > (1.0 / self.alpha):
            self.reject = True
            self.M = 1.0  
        else:
            self.reject = False
        return self.reject

class StreamingDriftDetector:
    def __init__(self, 
                 model_registry=None,
                 model_m='rf_regressor',
                 ref_batch_size=200,
                 burnin=5,
                 alpha=0.05,
                 fdr_method='LORD', 
                 martingale=False,
                 use_page_hinkley=True,
                 use_adwin=True,
                 use_hst=False):
        self.model_registry = model_registry
        self.model_m = model_m
        self.ref_batch_size = ref_batch_size
        self.burnin = burnin
        self.alpha = alpha
        self.fdr_method = fdr_method
        self.use_martingale = martingale
        self.use_ph = use_page_hinkley
        self.use_adwin = use_adwin
        self.use_hst = use_hst
        self.X_buffer = []
        self.y_buffer = []
        self.is_fitted = False
        self.model = None
        
        self.pval_calculator = ExponentialWeightedPValue(lam=0.95, burnin=burnin)
        self.fdr_controller = OnlineFDR(method=fdr_method, alpha=alpha)
        self.martingale = StreamingMartingale(alpha=alpha) if martingale else None
        
        self.ph = drift.PageHinkley(delta=0.005, threshold=50.0, min_instances=30) if use_ph else None
        self.adwin = drift.ADWIN(delta=0.02, clock=1, grace_period=30) if use_adwin else None
        self.hst = HalfSpaceTrees(n_trees=150, height=4, window_size=200) if use_hst else None
        
        self.mse_history = []
        self.pval_history = []
        self.rej_history = []
        self.drift_events = []
        
    def set_model(self, model):
        self.model = model
        
    def partial_fit_model(self, X, y):
        if self.model is None:
            return
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X, y)
        elif hasattr(self.model, 'warm_start'):
            self.model.warm_start = True
            self.model.fit(X, y)
        else:
            if len(self.X_buffer) < 5000:
                X_all = np.array(self.X_buffer)
                y_all = np.array(self.y_buffer)
                self.model.fit(X_all, y_all)
    
    def update(self, X, y):
        if not self.is_fitted:
            self.X_buffer.append(X)
            self.y_buffer.append(y)
            if len(self.X_buffer) >= self.ref_batch_size:
                if self.model is not None:
                    self.partial_fit_model(np.array(self.X_buffer), np.array(self.y_buffer))
                self.is_fitted = True
                self.X_buffer = self.X_buffer[-200:]
                self.y_buffer = self.y_buffer[-200:]
            return {'status': 'collecting', 'n': len(self.X_buffer)}
        
        X_arr = np.array([X]) if not isinstance(X, np.ndarray) else X
        y_pred = self.model.predict(X_arr)
        mse = np.mean((np.array([y]) - y_pred) ** 2)
        self.mse_history.append(mse)
        if len(self.mse_history) > 5000:
            self.mse_history = self.mse_history[-5000:]
        
        pval = self.pval_calculator.update(mse)
        self.pval_history.append(pval)
        if len(self.pval_history) > 5000:
            self.pval_history = self.pval_history[-5000:]
        
        fdr_reject = self.fdr_controller.update(pval)
        self.rej_history.append(fdr_reject)
        
        mart_reject = False
        if self.martingale:
            mart_reject = self.martingale.update(pval)
        
        ph_drift = False
        if self.ph:
            ph_drift = self.ph.update(mse)
        
        adwin_drift = False
        if self.adwin:
            self.adwin.update(mse)
            adwin_drift = self.adwin.drift_detected
        
        hst_drift = False
        if self.hst:
            score = self.hst.score_one({'mse': mse})
            self.hst.learn_one({'mse': mse})
            if len(self.mse_history) > 100:
                thresh = np.percentile(self.mse_history[-100:], 95) if len(self.mse_history) >= 100 else 1e9
                hst_drift = mse > thresh * 1.5 
        
        overall_drift = (fdr_reject or mart_reject or ph_drift or adwin_drift or hst_drift)
        if overall_drift:
            self.drift_events.append(len(self.mse_history)-1)
        
        if fdr_reject or ph_drift:
            if len(self.X_buffer) > 50:
                self.partial_fit_model(np.array(self.X_buffer[-50:]), np.array(self.y_buffer[-50:]))
        
        return {
            'mse': mse,
            'pval': pval,
            'fdr_reject': fdr_reject,
            'martingale_reject': mart_reject,
            'ph_drift': ph_drift,
            'adwin_drift': adwin_drift,
            'hst_drift': hst_drift,
            'overall_drift': overall_drift,
            'total_drift_events': len(self.drift_events)
        }
    
    def get_summary(self):
        return {
            'total_processed': len(self.mse_history),
            'drift_events': self.drift_events,
            'n_drifts': len(self.drift_events),
            'last_pval': self.pval_history[-1] if self.pval_history else None,
            'last_fdr_reject': self.rej_history[-1] if self.rej_history else None
        }
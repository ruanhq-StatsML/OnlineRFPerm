'''
Identifying the point that the distribution shift starts to become significant.
'''
from bayesian_changepoint_detection.online_changepoint_detection import (
    online_changepoint_detection,
    constant_hazard,
    StudentT)
from bocpd.bayesian_models import online_changepoint_detection as bocpd_online
from bocpd.online_likelihoods import StudentT as bocpd_studentT
from frouros.detectors.concept_drift import DDM, STEPD, HDDMA, ECDDWT
from river.anomaly import HalfSpaceTrees
from bocpd.bayesian_models import online_changepoint_detection as bocpd_online
from bocpd.online_likelihoods import StudentT as BocpdStudentT
from bocpd.hazard_functions import constant_hazard as bocpd_constant_hazard
from frouros.detectors.concept_drift import DDM, STEPD, HDDMA, ECDDWT
from skmultiflow.drift_detection import ADWIN as SKM_ADWIN
from skmultiflow.drift_detection import KSWIN as SKM_KSWIN
#from fitness_anomaly_detection import FitnessGaussian
from bocpd.bayesian_models import online_changepoint_detection as bocpd_online
from ocpdet import EWMA, TwoSample

def first_k_consecutive_rej(rej, k):
    n = len(rej)
    for i in range(n - k):
        if np.all(rej[i:(i+k)]):
            return (i+k)
    return np.nan

def benchmark_whole_onlinePermOOB(MSE_list, alpha = 0.05):
    result = {}
    #Bayesian Change-Point Detection(from the standard implementation)
    mu0 = float(np.mean(MSE_list[:min(50, len(MSE_list))]))
    obs = StudentT(alpha = 1.0, beta = 1.0, kappa = 1.0, mu = mu0)
    hazard = lambda r: constant_hazard(100, r)
    _, maxes = online_changepoint_detection(MSE_list, hazard, obs)
    det = np.zeros(len(MSE_list), dtype = bool)
    for i in range(50, len(maxes)):
        if maxes[i] < 3 or (i > 0 and maxes[i] < maxes[i - 1] - 15):
            det[i - 1] = True
    results['BOCPD_bcpd_2'] = first_k_consecutive_rej(det, 2)
    results['BOCPD_bcpd_1'] = first_k_consecutive_rej(det, 1)
    results['BOCPD_bcpd_3'] = first_k_consecutive_rej(det, 3)
    #EWMA:
    model = EWMA(r = 0.1, burnin = 25)
    model.process(MSE)
    change_point_list = model.changepoint
    results['EWMA_3'] = first_k_consecutive_rej(change_point_list, 3)        
    results['EWMA_2'] = first_k_consecutive_rej(change_point_list, 2)
    results['EWMA_1'] = first_k_consecutive_rej(change_point_list, 1)      
    model = TwoSample(statistic = 'Lepage', threshold = 2.5)
    model.process(MSE)
    change_point_list = model.changepoint
    results['TS_3'] = first_k_consecutive_rej(change_point_list, 3)        
    results['TS_2'] = first_k_consecutive_rej(change_point_list, 2)
    results['TS_1'] = first_k_consecutive_rej(change_point_list, 1)    
    #TwoSample:
    model = TwoSample(statistic = 'Lepage', threshold = 2.5)
    model.process(MSE)
    change_point_list = model.changepoint
    results['TS_3'] = first_k_consecutive_rej(change_point_list, 3)        
    results['TS_2'] = first_k_consecutive_rej(change_point_list, 2)
    results['TS_1'] = first_k_consecutive_rej(change_point_list, 1)
    #ADWIN:
    adwin = river_drift.ADWIN(delta = 0.02, clock = 1, grace_period = 30)
    det = []
    for MSE in MSE_list:
        adwin.update(MSE)
        det.append(adwin.drift_detected)
    results['ADWIN_3'] = first_k_consecutive_rej(det, 3)        
    results['ADWIN_2'] = first_k_consecutive_rej(det, 2)
    results['ADWIN_1'] = first_k_consecutive_rej(det, 1)
    #HDMMA:
    detector = SKM_ADWIN(delta = 0.05)
    det = []
    for MSE in MSE_list:
        detector.add_element(float(MSE))
        det.append(detector.detected_change())
    results["HDMMA_3"] = first_k_consecutive_rej(det, 3)
    results["HDMMA_2"] = first_k_consecutive_rej(det, 2)
    results["HDMMA_1"] = first_k_consecutive_rej(det, 1)
    #ECDDWT: 
    detector = SKM_KSWIN(window_size = 50, stat_size = 20)
    det = []
    for MSE in MSE_list:
        detector.add_element(float(MSE))
        det.append(detector.detected_change())
    results["ECDDWT_3"] = first_k_consecutive_rej(det, 3)
    results["ECDDWT_2"] = first_k_consecutive_rej(det, 2)
    results["ECDDWT_1"] = first_k_consecutive_rej(det, 1) 
    #FITNESS(Gaussian Stream Online Anomaly Detection)
    burnin_win = min(50, len(MSE_list)//4)   
    sigma = max(np.std(MSE_list[:burn_in_win]), 1e-8) if burnin_win > 1 else 1.0
    detector = FitnessGaussian(sigma = float(sigma), delta = 0.01)
    scores = []
    for r in MSE_list:
        s, _ = detector.update(np.array([float(r)], dtype = np.float32))
        scores.append(s)
    scores = np.array(scores)
    threshold = np.percentile(scores[:min(200, len(scores))], 95) if len(scores) >= 20 else (np.max(scores) if len(scores) > 0 else 0.0)
    det = [s > threshold for s in scores]
    results['FITNESS_1'] = first_k_consecutive_rej(det, 1)
    results['FITNESS_2'] = first_k_consecutive_rej(det, 2)
    results['FITNESS_3'] = first_k_consecutive_rej(det, 3)
    #HalfSpace Tree based methods, adapted from the original implementation
    init_len = min(30, len(MSE_list))
    mi, ma = float(np.min(MSE_list[:init_len])), float(np.max(MSE_list[:init_len]))
    if ma - mi < 1e-12:
        ma = mi + 1.0
    scaled = np.clip((MSE_list - mi) / (ma - mi), 0.0, 1.0)
    hst = HalfSpaceTrees(n_trees=150, height=4, window_size=200)
    scores = []
    for v in scaled:
        x = {"r": float(v)}
        scores.append(hst.score_one(x))
        hst.learn_one(x)
    threshold = np.percentile(scores[:min(200, len(scores))], 95) if len(scores) >= 20 else (np.max(scores) if scores else 0.0)
    det = [s > threshold for s in scores]
    results["HalfSpaceTrees_1"] = first_k_consecutive(det, 1)
    results["HalfSpaceTrees_2"] = first_k_consecutive(det, 2)
    results["HalfSpaceTrees_3"] = first_k_consecutive(det, 3)
    #Martingale based methods:
    pvals = empirical_pval(MSE_list)
    rej_martingale = method_martingale(pvals, alpha = alpha)
    rej_martingale2 = method_martingale2(pvals, alpha = alpha)
    results['martingale_1'] = first_k_consecutive_rej(rej_martingale, 1)
    results['martingale_2'] = first_k_consecutive_rej(rej_martingale, 2)
    results['martingale_3'] = first_k_consecutive_rej(rej_martingale, 3)
    results['martingale2_1'] = first_k_consecutive_rej(rej_martingale2, 1)
    results['martingale2_2'] = first_k_consecutive_rej(rej_martingale2, 2)
    results['martingale2_3'] = first_k_consecutive_rej(rej_martingale2, 3)
    rej_fix = (np.asarray(pvals) <= alpha)
    results['fix_1'] = first_k_consecutive_rej(rej_fix, 1)
    results['fix_2'] = first_k_consecutive_rej(rej_fix, 2)
    results['fix_3'] = first_k_consecutive_rej(rej_fix, 3)
    rej_addis = method_addis(pvals, alpha = alpha)
    rej_saffron = method_saffron(pvals, alpha = alpha)
    results['addis_1'] = first_k_consecutive_rej(rej_addis, 1)
    results['addis_2'] = first_k_consecutive_rej(rej_addis, 2)
    results['addis_3'] = first_k_consecutive_rej(rej_addis, 3)
    results['saffron_1'] = first_k_consecutive_rej(rej_saffron, 1)
    results['saffron_2'] = first_k_consecutive_rej(rej_saffron, 2)
    results['saffron_3'] = first_k_consecutive_rej(rej_saffron, 3)      
    return results






















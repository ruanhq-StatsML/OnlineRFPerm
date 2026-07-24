#rrcf benchmark:
import os
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
#from model_registry_class import ModelRegistry
from river import drift
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
from frouros.detectors.concept_drift import DDM, STEPD, HDDMA, ECDDWT
import numpy as np
import scipy
os.chdir('/Users/heqiaoruan/Documents/Github/OnlineExtension_RFPerm/Python')
#from onlinePermOOB import onlinePermOOB_wholedf
from benchmark_methods import *
from model_registry_class import ModelRegistry
from scipy.io import arff
import pandas as pd
import rrcf









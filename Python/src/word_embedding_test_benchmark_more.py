#Word Embedding Benchmarks:
import os
os.chdir('/Users/heqiaoruan/Documents/Github/SAPerm')
from model_registry_class import ModelRegistry
import numpy as np
from scipy.stats import f, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from benchmark_method_tst import (
    chen_qin, adaptive_test_high_dim,
    c2st, zwl_test, bai96_test,
    sri08_test, skk_test,
    random_projection_test,
    kernel_MMD, chen_li14
)
from causal_testing_utils import *
from numba import njit, prange
from CFPerm_VIMP_utils import *
from RFPerm import RFPerm
from conformal_benchmark import *
#Model Configuration of the learner.
from model_registry_class import *
model_factory = ModelRegistry(
ntree = 150, ridge_alpha = 0.25,
nthread = 1, maxit = 500,
max_depth = 5, gamma = 0.25,
eta = 0.15, mlp_hidden_size = 4,
mlp_decay = 1e-4, mlp_max_iter = 500
)
model_registry = model_factory.as_r_style_dict()
func_config = {
    'n_perm_cf': 150,
    'n_perm_rf': 150,
    'n_perm_drperm': 150,
    'n_perm_rrperm': 150,
    'outcome_model_causal': 'rf_regressor',
    'ps_model': '',
    'kernel': 'rbf',
    'n_conformal': 150,
    'ps_model': 'logistic_classifier'
}

def L2(X, Y):
    np.mean((np.array(X) - np.array(Y))**2)

def benchmark_all_method(df1, df2, B = 150, model_registry = model_registry, config = func_config):
n1, p = df1.shape
n2 = df2.shape[0]
df_batch1 = np.asarray(df1[:,:(df1.shape[1]-1)], dtype=float)
df_batch2 = np.asarray(df2[:,:(df2.shape[1]-1)], dtype=float)
if df_batch1.ndim == 1:
    df_batch1 = df_batch1.reshape(-1, 1)
if df_batch2.ndim == 1:
    df_batch2 = df_batch2.reshape(-1, 1)
Y1 = df1[:, -1]
Y2 = df2[:, -1]
Y = np.concatenate([Y1, Y2]).reshape(-1,1)
W = np.concatenate([np.zeros(n1, dtype = int),
                np.ones(n2, dtype = int)])
n_perm_cf = config.get("n_perm_cf")
n_perm_rf = config.get('n_perm_rf')
n_perm_drperm = config.get('n_perm_drperm')
outcome_model = config.get('outcome_model_causal')
kernel = config.get('kernel')
n_perm_rrperm = config.get('n_perm_rrperm')
ps_model = config.get('ps_model')
n_conformal = config.get('n_conformal')
#Specify the Outcome Model and Propensity Score Model:
model_m = model_registry[outcome_model]
model_e = model_registry[ps_model]
#Initiate the p-value lists, for the two sample testing procedure, run on the joint distribution as well as the covariate distributions
p_val_rrperm = np.zeros(B)
p_val_miles16 = np.zeros(B)
p_val_bai96 = np.zeros(B)
p_val_xu16 = np.zeros(B)
p_val_chen10 = np.zeros(B)
p_val_chen14 = np.zeros(B)
p_val_c2st = np.zeros(B)
p_val_kmmd = np.zeros(B)
p_val_zwl = np.zeros(B)
p_val_zwlm = np.zeros(B)
p_val_sri = np.zeros(B)
p_val_pan14 = np.zeros(B)
p_val_drperm = np.zeros(B)
p_val_cfperm_loco = np.zeros(B)
p_val_cfperm_permu = np.zeros(B)
p_val_cfperm_grf = np.zeros(B)
p_val_rfperm = np.zeros(B)
p_val_autotst = np.zeros(B)
p_val_skk = np.zeros(B)
p_val_xgbperm = np.zeros(B)
p_val_conformal = np.zeros(B)
#Resampling via the Bootstrap procedure:
for i in prange(B):
Y_res1 = Y1.reshape(-1, 1).astype(float)
Y_res2 = Y2.reshape(-1, 1).astype(float)
df1_concat = np.concatenate([df_batch1, Y_res1], axis = 1).astype(float)
df2_concat = np.concatenate([df_batch2, Y_res2], axis = 1).astype(float)
df1_X_concat = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
df2_X_concat = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
#Bootstrap version of the bootstrap sample:
df1_sample = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
df2_sample = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
X = np.concatenate([df1_sample[:, :(df1_sample.shape[1] - 1)], 
                    df2_sample[:, :(df2_sample.shape[1] - 1)]]).astype(float)
Y = np.concatenate([df1_sample[:, df1_sample.shape[1] - 1],
                    df2_sample[:, df2_sample.shape[1] - 1]]).astype(float)
    p_val_rrperm[i] = RRPerm(X, Y, W, n_perm = n_perm_rrperm, model_m = model_m, model_e = model_e)
    p_val_cfperm_loco[i] = cfperm_feature_pvals(X, Y, W,
     model_registry = MODEL_REGISTRY,
     vimp = 'loco', n_perm = n_perm_cf,
        model_m = model_m, model_e = model_e)
    p_val_cfperm_permu[i] = cfperm_feature_pvals(X, Y, W,
        model_registry = MODEL_REGISTRY,
        vimp = 'permucate', n_perm = n_perm_cf,
        model_m = model_m, model_e = model_e)
    p_val_cfperm_grf[i] = cfperm_feature_pvals(X, Y, W,
        model_registry = MODEL_REGISTRY,
        vimp = 'grf', n_perm = n_perm_cf,
        model_m = model_m, model_e = model_e)
    p_val_drperm[i] = DRPerm(X, Y, W,
        model_registry = MODEL_REGISTRY,
        n_perm = n_perm_drperm,
        model_m = model_m, model_e = model_e)
    p_val_miles16[i] = random_projection_test(df1_X_concat, df2_X_concat)[1]
    p_val_xu16[i] = list(adaptive_test_high_dim(df1_X_concat, df2_X_concat)[1].values())
    p_val_kmmd[i] = kernel_MMD(df1_X_concat, df2_X_concat, kernel = kernel)
    p_val_chen14[i] = chen_qin(df1_X_concat, df2_X_concat)
    p_val_chen10[i] = chen_li14(df1_X_concat, df2_X_concat, n_perm = 100)[0]
    p_val_pan14[i] = pan14(df1_X_concat, df2_X_concat)[1]
    p_val_skk[i] = skk_test(df1_X_concat, df2_X_concat)[1]
    p_val_sri[i] = sri08_test(df1_X_concat, df2_X_concat)
    p_val_bai96[i] = bai96_test(df1_X_concat, df2_X_concat)
    p_val_zwl[i] = zwl_test(df1_X_concat, df2_X_concat, order = 1)[1]
    p_val_zwlm[i] = zwl_test(df1_X_concat, df2_X_concat, order = 2)[1]
    p_val_c2st[i] = c2st(df1_X_concat, df2_X_concat)
    p_val_rfperm[i] = RFPerm(pd.DataFrame(df1_sample), pd.DataFrame(df2_sample), loss = L2, B = n_perm_rf)
    p_val_xgbperm[i] = PermValTest(df1_sample, df2_sample, model_class = model_class, model_component = model_component_xgb)
    p_val_autotst[i] = autotst.AutoTST(df1_X_concat, df2_X_concat).p_value()
    p_val_conformal[i] = conformalNN(df1_sample, df2_sample, mc = n_conformal)
#Reformulate them back to a DataFrame:
out_df = pd.DataFrame({
    'xu16': float(np.mean(p_val_xu16 < alpha)),
    'pan14': float(np.mean(p_val_pan14 < alpha)),
    'chen10': float(np.mean(p_val_chen10 < alpha)),
    'chen14': float(np.mean(p_val_chen14 < alpha)),
    'c2st': float(np.mean(p_val_c2st < alpha)),
    'sri': float(np.mean(p_val_sri < alpha)),
    'skk': float(np.mean(p_val_skk < alpha)),
    'zwl': float(np.mean(p_val_zwl < alpha)),
    'zwlm': float(np.mean(p_val_zwlm < alpha)),
    'bai96': float(np.mean(p_val_bai96 < alpha)),
    'rrperm': float(np.mean(p_val_rrperm < alpha)),
    'drperm': float(np.mean(p_val_drperm < alpha)),
    'miles16': float(np.mean(p_val_miles16 < alpha)),
    'kmmd': float(np.mean(p_val_kmmd < alpha)),
    'cfperm_loco': float(np.mean(p_val_cfperm_loco)),
    'cfperm_permu': float(np.mean(p_val_cfperm_permu)),
    'cfperm_grf': float(np.mean(p_val_cfperm_grf)),
    'conformal': float(np.mean(p_val_conformal)),
    'rfperm': float(np.mean(p_val_rfperm < alpha)),
    'xgbperm': float(np.mean(p_val_xgbperm < alpha)),
    'autotst': float(np.mean(p_val_autotst < alpha))
    })
    return out_df


df1 = np.random.random((200, 21))
df2 = np.random.random((200, 21))




























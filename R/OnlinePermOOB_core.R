#Performing the Online PermOOB Procedure for documentation, readily available for R package.
pacman::p_load(RegimeChange, nnet, keras, tidymodels, tidyverse, tidytable)
source('model_registry.R')
#Empirical p-value procedure here.
empirical_pval <- function(MSE_dif_list, burnin = 5){
  n = length(MSE_dif_list)
  pval_list <- numeric(n)
  pval_list[1:burnin] <- 1
  for(i in (burnin+1):n){
    pval_list[i] <- (1.0 + sum(MSE_dif_list[burnin:(i-1)] > MSE_dif_list[i]))/(i - burnin)
  }
  return(pval_list)
}

#Gamma sequence by default.
gamma_seq <- function(n){
  return(1/c(1:n))
}


#Online FDR control procedure with the recent discoveries(significance).
online_fdr_lord <- function(pvals, alpha = 0.05){
  n <- length(pvals)
  gamma = gamma_seq(n)
  alpha_t <- numeric(n)
  reject <- logical(n)
  rej_times <- integer(0)
  W <- alpha
  for(t in seq_len(n)){
    if(is.na(pvals[t]) || W <= 1e-12){
      next
    }
    a <- gamma[t] * alpha
    if(length(rej_times) > 0){
      for(tau in rej_times){
        lag <- t - tau
        if(lag > 0 && lag <= n){
          a <- a + gamma[lag] * alpha
        }
      }
    }
    alpha_t[t] <- min(a, W)
    reject[t] <- (pvals[t] <= alpha_t[t])
    #Update the wealth after this step:
    W <- W - alpha_t[t] + ifelse(reject[t], alpha, 0.0)
    if(reject[t]){
      rej_times <- c(rej_times, t)
    }
  }
  return(list(alpha_t = alpha_t, reject = reject))
}

#Online FDR control procedure with the number of the recent discoveries(significance).
online_fdr_lond <- function(pvals, alpha = 0.05){
  n <- length(pvals)
  gamma <- gamma_seq(n)
  alpha_t <- numeric(n)
  reject <- logical(n)
  nr <- 0
  for(t in seq_len(n)){
    if(is.na(pvals[t])){
      next
    }
    alpha_t[t] <- alpah * gamma[t] * (1 + nr)
    reject[t] <- (pvals[t] <= alpha_t[t])
    if(reject[t]){
      nr <- nr + 1
    }
  }
  return(list(alpha_t = alpha_t, reject = reject))
}

#Quantile Aggregation on the batched MSE to give people information beyond the mean.
quantile_aggregation <- function(MSE, q_list, w){
  agg_MSE <- 0
  if(sum(w) == 0){
    return(NA)
  }
  w <- w/sum(w)
  for(i in 1:length(q_list)){
    agg_MSE <- agg_MSE + unname(quantile(MSE, q_list[i])) * q_list[i] * w[i]
  }
  return(agg_MSE)
}

#Exponential Weighted Moving Average p-value for a specific time-point.
ewma_pval <- function(MSE_list, MSE_new, lambda = 0.9, smoother = 0.005){
  t <- length(MSE_list)
  if(t == 0){
    return(1.0)
  }
  MSE_list <- as.numeric(MSE_list)
  arr <- seq_len(t)
  weights <- exp(-lambda * (t - arr))
  weights_sum <- sum(weights)
  if(weights_sum <= 0){
    return(1.0)
  }
  weights <- weights/weights_sum
  weights_count <- sum((MSE_list >= MSE_new) * weights)
  return((smoother+ weights_count)/(smoother + weights))
}

#Exponential Weighted Moving Average p-value for a list of the MSE(mean-squared-errors).
ewma_pvalue <- function(MSE_list, lambda = 0.9, burnin = 1){
  n <- length(MSE_list)
  pvals <- numeric(n - burnin)
  for(t in seq_len(n - burnin)){
    idx <- t + burnin
    pvals[t] <- ewma_pval(MSE_list[(burnin+1):(idx - 1)],
        MSE_list[idx], lambda = lambda)
  }
  return(pvals)
}

#The fixed alpha procedure in comparison with the p-value.
method_fix <- function(pval_list, alpha){
  return((pval_list < alpha))
}

##############################
#The martingale/e-value based methods for the online anomaly detection.
method_martingale2 <- function(pvals, alpha = 0.05, eps = 1e-8, lam = 4) {
  pvals <- pmax(pmin(as.numeric(pvals), 1.0), eps)
  evals <- lam / (exp(lam) - 1) * exp(lam * (1 - pvals))
  cumprod(evals) > (1.0 / alpha)
}

method_martingale3 <- function(pvals, alpha = 0.05, eps = 1e-10) {
  pvals <- pmax(pmin(as.numeric(pvals), 1.0), eps)
  evals <- (2.0 / 3.0) * pvals^(-1.0 / 3.0)
  cumprod(evals) > (1.0 / alpha)
}

method_martingale4 <- function(pvals, alpha = 0.05, eps = 1e-10) {
  pvals <- pmax(pmin(as.numeric(pvals), 1.0), eps)
  evals <- (4.0 / 5.0) * pvals^(-0.2)
  cumprod(evals) > (1.0 / alpha)
}

method_martingale <- function(pvals, alpha = 0.05, eps = 1e-10) {
  pvals <- pmax(pmin(as.numeric(pvals), 1.0), eps)
  evals <- 1.0 / (2.0 * sqrt(pvals))
  M <- cumprod(evals)
  M > (1.0 / alpha)
}

#e-value procedure for the online FDR control 
method_evalue <- function(pvals, alpha = 0.05) {
  pvals <- as.numeric(pvals)
  e_values <- 1.0 / pmax(pvals, 1e-12)
  wealth <- 1.0
  rej <- logical(length(pvals))
  threshold <- 1.0 / alpha
  for (i in seq_along(e_values)) {
    wealth <- wealth * e_values[i]
    if (wealth >= threshold) {
      rej[i] <- TRUE
      wealth <- 1.0
    }
  }
  rej
}

####################################################################################################
# -------------------- SAFFRON and ADDIS procedures --------------------                           #
#Online Multiple testing procedures:                                                               #
#SAFFRON for the Online FDR Control                                                                #
#Serial estimate of the Alpha Fraction that is Futilely Rationed On true Null hypotheses           #
#https://arxiv.org/pdf/1802.09098
####################################################################################################
method_saffron <- function(pvals, alpha = 0.05, lam = 0.3, tau = 0.5, s = 1.0) {
  n <- length(pvals)
  pvals <- as.numeric(pvals)
  g <- (1 / (1:n)) / sum(1 / (1:n))   # normalized harmonic weights
  nc <- 0
  W <- alpha
  rej <- logical(n)
  for (i in seq_len(n)) {
    if (is.na(pvals[i])) next
    if (pvals[i] <= lam || pvals[i] > tau) {
      W <- W + tau
      next
    }
    nc <- nc + 1
    a <- min((tau - lam) * g[nc] * W, W)
    rej[i] <- (pvals[i] <= a)
    W <- W - a + (if (rej[i]) alpha else 0.0)
  }
  rej
}

#ADDIS(ADaptive algorithm that DIScards conservative nulls) for the Online FDR Control: 
#https://arxiv.org/abs/1905.11465
method_addis <- function(pvals, alpha = 0.05, lam = 0.045, tau = 0.3, s = 1.0) {
  n <- length(pvals)
  pvals <- as.numeric(pvals)
  g <- (1 / (1:n)) / sum(1 / (1:n))
  W <- alpha
  nc <- 0
  rej <- logical(n)
  for (i in seq_len(n)) {
    if (is.na(pvals[i])) next
    if (pvals[i] <= lam || pvals[i] > tau) {
      W <- W + tau
      next
    }
    nc <- nc + 1
    a <- min((tau - lam) * g[nc] * W, W)
    rej[i] <- (pvals[i] <= a)
    W <- W - a + (if (rej[i]) alpha else 0.0)
  }
  rej
}

#Find the first indice in the series that starts the k consecutive rejections.
first_k_consecutive_rej <- function(rej, k){
  n <- length(rej)
  for(i in 1:(n - k + 1)){
    if(all(rej[i:(i+k-1)])){
      return(i)
    }
  }
  return(NA)
}

#Find the first indice in the series that starts the k consecutive rejections.
first_k_consecutive_rej_ind <- function(rej, k){
  n <- length(rej)
  for(i in 1:(n - k+1)){
    if(sum(rej[i:(i+k-1)]) == k){
      return(i)
    }
  }
  return(NA)
}

#The page-hinkley procedure for the online anomaly detection.
PageHinkley <- function(delta = 0.005, threshold = 50.0,
    min_instances = 30){
  n <- 0
  x_sum <- 0.0
  PH_n <- 0.0
  PH_min <- 0.0
  update <- function(x){
    n <- n + 1
    x_sum <- x_sum + x
    m_n <- x_sum/n
    PH_n <- PH_n + (x - m_n - delta)
    PH_min <- min(PH_min, PH_n)
    if(n < min_instances){
      return(FALSE)
    }
    return((PH_n - PH_min) > threshold)
  }
  return(list(update = update))
}



#model_spec should be other columns or other aspects here.
#The method include "Unweighted" and "EWMA"(Exponential Weighted Moving Average Procedure)
############################################################
#' Online PermOOB procedure - under the regression settings here.
#'
#' This function conduct the online-permoob procedure via the online anomaly detection procedure
#' through an empirical p-value based procedure, via a model based procedure on the model's predictive performance
#'
#' @param data:           A data.frame, the last column should be the continous response, with the column name defined.
#' @param model_spec:     The model specification, with the proper hyperparameter setting, in practice, hyperparameter tuning is not recommended.
#' @param ref_batch_size: The reference batch-size for the initial model, by default = 100, at least 100.
#' @param batch_size:     The batch_size for the evaluation stage, by default = 10.
#' @param method_p:       The aggregation procedure for the empirical p-values.
#' @param burnin:         The size of the burn-in periods, by default, 0 - burnin-period is not really needed for achieving the desirable performance.
#' @param alpha:          The significance level, by default = 0.05. 
#' @param QA:             Whether to conduct the Quantile Aggregation Procedure - with the quantile weighted, need batch_size >= 10 and the multiplier of 10, in practice, the unweighted procedure + EWMA would be good enough.
#' @param seed:           The random seed, by default = 2026
#' @param q_list:         The quantile list for that in the quantile aggregation procedure.
#' @param w:              The weights for the quantile aggregation procedure
#' @param lam_saffron:    The lambda paramater in the saffron procedure, controls the conservativenesss of SAFFRON Online-FDR Procedure.
#' @param tau_saffron:    The tau paramater in the saffron procedure, controls the conservativenesss of SAFFRON Online-FDR Procedure.
#' @param lam_addis:      The lambda paramater in the addis procedure, controls the conservativenesss of ADDIS Online-FDR Procedure.
#' @param tau_addis:      The tau paramater in the addis procedure, controls the conservativenesss of ADDIS Online-FDR Procedure.
#' @param train_prop:     The proportion of the training data in the reference batch size.
#' @param formula:        The fitting formula of the data, by default 'y~.' - means regress response y column(the name should be aligned) to all of the other columns(covariates)
#' 
###############################################################
onlinePermOOB_regression_prototype <- function(data, model_spec,
    ref_batch_size = 2000, batch_size = 20, method_p = 'Unweighted', 
    burnin = 0, alpha = 0.05, QA = FALSE, seed = 2026,
    q_list = seq(0.1, 0.9, 0.1), w = rep(1, 9), 
    lam_saffron = 0.3, tau_saffron = 0.5,
    lam_addis = 0.045, tau_addis = 0.3, 
    train_prop = 0.7, formula = y~.){
  ref_data <- data[1:ref_batch_size, ]
  stream_data <- data[(ref_batch_size+1):(nrow(data)), ]
  wt <- workflow() %>% add_model(model_spec) %>% add_formula(formula)
  set.seed(seed)
  pert_idx <- sample(c(1:nrow(ref_data)), nrow(ref_data), replace = TRUE)
  tr_idx <- pert_idx[1:round(nrow(ref_data) * train_prop)]
  va_idx <- pert_idx[(round(nrow(ref_data) * train_prop) + 1):nrow(ref_data)]
  ref_data_train <- ref_data[tr_idx, ]
  ref_data_val <- ref_data[va_idx, ]
  set.seed(seed + 1)
  fitted_wt <- wt %>% fit(data = ref_data_train)
  n_stream <- nrow(stream_data)
  if(n_stream <= 0){
    return(tibble(batch_id = integer(), mse = numeric()))
  }
  batch_ids <- unique(ceiling(seq_len(n_stream)/batch_size))
  #That's exactly you want to develop this procedure for the evaluation & prediction:
  original_pre <- fitted_wt %>% predict(new_data = ref_data_val)
  original_pred <- original_pre$.pred
  original_mse <- mean((original_pred - ref_data_val$y) ** 2)
  #rolling pre:
  results <- map_dfr(batch_ids, function(b, QA = FALSE){
  	set.seed(seed + b + 10)
    start_idx <- (b-1) * batch_size + 1
    end_idx <- min(b * batch_size, n_stream)
    batch_data <- stream_data[start_idx:end_idx, ] 
    preds <- fitted_wt %>% predict(new_data = batch_data)
    y <- batch_data$y
    if(QA){
      mse_list = (preds$.pred - y) ** 2
      mse_val = quantile_aggregation(mse_list, q_list, w)
    }
    else{
      mse_val = mean(preds$.pred - y) ** 2
    }
    tibble(batch_id = b, mse = mse_val, n_obs = nrow(batch_data))
    })
  mse_dif_list <- results$mse - original_mse
  if(method_p == 'Unweighted'){
    pval_list <- empirical_pval(mse_dif_list, burnin = burnin)
  }
  else{
    pval_list <- ewma_pvalue(mse_dif_list, burnin = burnin)
  }
  pval_list <- empirical_pval(mse_dif_list, burnin = burnin)
  rej_addis <- method_addis(pval_list, lam = lam_addis, tau = tau_addis)
  rej_saffron <- method_saffron(pval_list, lam = lam_saffron, tau = tau_saffron)
  sum_addis <- sum(rej_addis)
  sum_saffron <- sum(rej_saffron)
  start_ind1_addis <- first_k_consecutive_rej(rej_addis, k = 1)
  start_ind2_addis <- first_k_consecutive_rej(rej_addis, k = 2)
  start_ind3_addis <- first_k_consecutive_rej(rej_addis, k = 3)
  start_ind1_saffron <- first_k_consecutive_rej(rej_saffron, k = 1)
  start_ind2_saffron <- first_k_consecutive_rej(rej_saffron, k = 2)
  start_ind3_saffron <- first_k_consecutive_rej(rej_saffron, k = 3)
  rej_fix <- method_fix(pval_list, alpha = alpha)
  sum_fix <- sum(rej_fix)
  start_ind1_fix <- first_k_consecutive_rej(rej_fix, k = 1)
  start_ind2_fix <- first_k_consecutive_rej(rej_fix, k = 2)
  start_ind3_fix <- first_k_consecutive_rej(rej_fix, k = 3)
  output = list(
    method = method_p,
    mse_ref = original_mse,
    mse_stream = results$mse,
    rej_addis = rej_addis,
    rej_saffron = rej_saffron,
    rej_fix = rej_fix, 
    sum_addis = sum_addis,
    start_ind1_addis = start_ind1_addis,
    start_ind2_addis = start_ind2_addis,
    start_ind3_addis = start_ind3_addis,
    sum_saffron = sum_saffron,
    start_ind1_saffron = start_ind1_saffron,
    start_ind2_saffron = start_ind2_saffron,
    start_ind3_saffron = start_ind3_saffron,
    sum_fix = sum_fix,
    start_ind1_fix = start_ind1_fix,
    start_ind2_fix = start_ind2_fix,
    start_ind3_fix = start_ind3_fix
  )
  return(output)
}





'''
#Simulating the Stationary Data-Generating Process, should reject nothing via the OnlineFDR procedure.
set.seed(2026)
df_X <- matrix(rnorm(10000, 0, 1), nrow = 1000, ncol = 10)
df_Y <- rnorm(1000)
df_input <- data.frame(cbind(df_X, df_Y))
colnames(df_input) <- c(paste("X", c(1:ncol(df_X)), sep = "_"), "y") #rename the last(response) column as y
rf_spec <- rand_forest(mtry = 5, trees = 100) %>%
  set_engine('ranger', importance = 'impurity') %>%
  set_mode('regression')
onlinePermOOB_regression_prototype(df_input, rf_spec, 
  ref_batch_size = 200, batch_size = 5, method_p = 'Unweighted',
   burnin = 0, train_prop = 0.6)

- Did not reject anything here via the OnlineFDR procedure but for the fixed alpha procedure, do generate some rejections.

$start_ind1_addis
[1] NA

$start_ind2_addis
[1] NA

$start_ind3_addis
[1] NA

$start_ind1_saffron
[1] NA

$start_ind2_saffron
[1] NA

$start_ind3_saffron
[1] NA

$start_ind1_fix
[1] 24

$start_ind2_fix
[1] 248

$start_ind3_fix
[1] 314
'''











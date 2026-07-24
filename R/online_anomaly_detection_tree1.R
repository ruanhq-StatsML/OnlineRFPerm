#online anomaly detection via the tree:


current_list_node <- function(df_stream, n_new, n_old){
  
}



#
'''

History L timestamps -> Future T timestamps illustration on the basic linear model:
LTSF-linear as a baseline for comparison:
\hat{X_{i}} = WX_{i} where W \in R^{T*L} is a linear layer along the temporal axis.
Are the prediction and input for each i_{th} variate

LTSF-Linear: set of linaer models;

SOTA FEDformer -> improvments by multivariate forecasting of ETT datasets in the Appendi


What can be learned for long-term forecasting:
(1) Temporal Dyamics in the look-back window significanctly impact the for
(2) AutoForemr ---> periodicity well only here.

generate_test_data <- function(n, p = 100){
  x <- matrix(rnorm(n * p), ncol = p)
  y <- 0.5 * x[, 1] + rnorm(n, sd = 1.5)
  data <- as.data.frame(cbind(y, x))
  colnames(data) <- c("y", paste0("x", 1:p))
  return(data)
}

Conducting the R coding:

generate_test_data <- function(n, p = 1200){
  x <- matrix(rnorm(n * p), ncol = p)
  y <- 0.5 * x[, 1] + rnorm(n, sd = 1.5)
  data <- as.data.frame(cbind(y, x))
  colnames(data) <- c("y", paste0("x", 1:p))
  return(data)
}

Operation on the sliding windows here:



'''










my_kmeans <- function(X, centers, max_iter = 100, 
    tol = 1e-4, nstart = 1){
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  if(length(centers) == 1){
  	K <-  centers
  	best_result <- NULL
  	best_ss <- Inf
  	for(start in 1:nstart){
  	  set.seed(start * 123)
  	  idx <- sample(n, K, replace = FALSE)
  	  current_centers <- X[idx, ]
  	  for(iter in 1:max_iter){
  	  	dist_sq <- rowSums(X^2) + t(rowSums(current_centers^2)) - 2 * X %*% t(current_centers)
  	  	cluster <- max.col(-dist_sq, ties.method = 'first')

  	  }
  	  if(nrow(new_centers) < K){
  	  	missing <- setdiff(1:K, unique(cluster))
  	  	dist_to_all_centers <- rowSums((X - current_centers[cluster,])^2)

  	  }
  	}
  }
}
dist_sq_final <- rowSums(X^2) + t(rowSums(current_centers^2)) -
2 * X %*% t(current_centers)

dist_sq_final <- rowSums(X^2) + t(rowSums(current_centers^2)) - 
2 * X %*% t(current_centers)


dist_sq_final <- rowSums(X^2) + t(rowSums(current_centers^2)) - 
2 * X %*% t(current_centers)

if(withinss < best_ss){
  best_ss <- withinss
  best_result <- list(cluster = cluster, centers = current_centers,
    iter = iter, withinss = withinss)
}

theta <- theta - learning_rate * grad
grad <- (1/length(idx)) * t(X_batch) %*% (h - y_batch)


sigmoid <- function(z){
  z <- pmax(pmin(z, 30), -30)
  return(1/(1+exp(-z)))
}

loss <- -mean(y_batch * log(h + 1e-8) + 
	(1 - y_batch) * log(1 - h + 1e-8))

logits <- cbind(1, X) %*% true_theta
probs <- 1/(1 + exp(-logits))
y <- rbinom(n, 1, prob)


for(epoch in 1:nepoch){
  if(is.null(batch_size)){
  	idx <- c(1:n)
  }
  else{
    idx <- sample(n, batch_size, replace = FALSE)
  }
  X_batch <- X[idx,]
  y_batch <- y[idx]
  linear <- X_batch %*% theta
  h <- sigmoid(linear)
  loss <- -mean(y_batch * log(h + 1e-8) + 
  	(1 - y_batch) * log(1 - h + 1e-8))
  loss_history <- c(loss_history, loss)
  if(length(loss_history) > 1 && abs(loss_history[epoch] - loss_history[epoch - 1]) < tol){
  	if(verbose){
  		cat('Converged at epoch:', epoch, '\n')
  		break
  	}
  }
  grad <- (1/length)
}















































D <- matrix(rnorm(1000, 0, 5), 100, 10)
which.max(D, dim = 1)

loss_hist <- c(loss_hist, loss)
grad <- (t(Xb) %*% (h - yb))/length(idx)


D2 <- matrix(rnorm(1000, 0, 5), 100, 10)


df <- data.frame(a = 1:5, b = 6:10, c = 11:15)
coords <- cbind(c(2, 4, 5), c(3, 1, 2))
df[coords] <- NA

xx <- rnorm(100)
c0 = c()
for(x in xx){
  c0 <- c(c0, x^2)
}
c0[1:10]

df <- data.frame(a = c(1:5), b = c(6:10), c = c(11:15))
coords <- cbind(c(1, 2, 4), c(1,2,3))
df[coords]#The indices are (1,1), (2,2), (4,3) indices






dist_sq <- rowSums(X_test ^2) + t(rowSums(X_train^2)) - 2 * X_test %*% t(X_train)
#for each of the row here!


dist_sq <- rowSums(X_test^2) + t()


dist_sq <- rowSums(X_test^2) + t(rowSums(X_train^2)) - 2 * X_test %*% t(X_train)
neigh_idx <- t(apply(dist_mat, 1, function(row){order(row)[1:k]})#Finding that indices
	#THen map them:
	dists <- dist_mat[i, idx]

order(row)

dist_sq <- rowSums(X_test^2) + t(rowSums(X_train^2)) - 2 * X_test %*% t(X_train)
dist <- sqrt(pmax(dist_sq, 0))
pred <- vector(mode = ifelse(type = 'class', 'character', 
'numeric'), length = n_test)
pred <- vector(mode = ifelse(type = 'class', 'character',
'numeric'), length = n_test)

neigh_y <- y_train[idx]

dist_sq <- rowSums(X_test^2) + t(rowSums(X_train^2)) - 2 * X_test %*% t(X_train)
names(which.max(tab))
tab <- table(neigh_y)

pred[i] <- names(which.max(weighted_tab))
w <- 1/(dist[i, idx] + 1e-9)
pred[i] <- weighted.mean(neigh_y, w)

pred[i] <- namess(which.max(weighted_tab))
w <- 1/(dist[i, idx] + 1e-9)
pred[i] <- weighted.mean(neigh_y, w)

gamma_seq <- function(n){
  gamma <- 1/c(1:n)
  gamma <- gamma/sum(gamma)
}

gamma_seq(10)





sim_empl



'''


mu = r(1-p)/p
sigma^2 = r(1-p)/(p**2 )

Technical mindset -> data fluency here 
Design the appropriate metrics, which statistical methods to leverage?
Validate a certain hypothesis here
'''

wt <- workflow() %>% add_model(model_spec) %>% add_formula(formula)
set.seed(seed)
fitted_wt <- wt %>% fit(data = ref_data_train)
pred_values <- fitted_wt %% predict(new_data = ref_data_val)
pred_val <- pred_values$.pred
#ungroup here
#.summarise() %>% gscore ->
#ungoropu



wt <- workflow() %>% add_model(model_spec) %>% 


wt <- workflow() %>% add_model(model_spec) %>% add_formula(formula)
set.seed(seed)
fitted_wt <- wt %>% fit(data = ref_data_train)
batch_ids <- unique(ceiling(seq_len(n_stream)/batch_size))

original_pre <- fitted_wt %>% predict(new_data = ref_data_val)
original_pred <- original_pre$.pred

preds <- fitted_wt %>% predict(new_data = batch_data)

preds <- fitted_wt %>% predict(new_data = batch_data)

y <- batch_data$y

fitted_wt <- wt %>% fit(data = ref_data_train)
batch_ids <- unique(ceiling(seq_len(n_stream)/batch_size))
original_pre <- fitted_wt %>% predict(new_data = ref_data_val)
original_pred <- original_pre$.pred
original_mse <- mean((original_pred - ref_data_val$y) ** 2)

original_pre <- fitted_wt %>% predict(new_data = ref_data_val)
original_pred <- original_pre$.pred
original_mse <- mean((original_pred - ref_data_val$y) ** 2)
mse_dif_list <- results$mse - original_mse

mse_dif_list <- results$mse - original_mse
original_pre <- fitted_wt %>% predict(new_data = ref_data_val)
original_pred <- original_pre$.pred

rbeta(n, )

rf_spec <- rand_forest(mtry = 5, trees = 100) %>%
  set_engine('ranger', importance = 'impurity') %>%
  set_mode('regression')

rf_spec <- rand_forest(mtry = 5, trees = 100) %>%
  set_engine('ranger', importance = 'impurity') %>%
  set_mode('regression')


xgb_spec <- boost_tree(mtry = 5, max_depth = 5) %>%
  set_engine('xgboost', importance = 'impurity') %>%
  set_mode('regression')


xgb_spec <- boost_tree(mtry = 5, max_depth = 4) %>%
  set_engine('xgboost', importance = 'impurity') %>%
  set_mode('regression')

rf_spec <- rand_forest(mtry = 5, trees = 100) %>%
  set_engine('ranger', importance = 'impurity') %>%
  set_mode('regression')



mlp_spec_nnet <- mlp(hidden_units = 10) %>%
  set_engine('nnet')


mlp_spec_keras <- mlp(hidden_units = 10, epochs = 20) %>%
  set_engine('keras')


mlp_spec_keras <- mlp(hidden_units = 10, epochs = 30) %>%
  set_engine('keras')



library(tidymodels)
mlp_spec <- mlp(
    hidden_units = 10,
    penalty = 0.01,
    epochs = 100
) %>% set_mode('regression') %>%
set_engine('nnet')
df <- data.frame(matrix(rnorm(11000), 1000, 11))
colnames(df) <- c(paste("X", c(1:10), sep = ""), "Y")
model0 <- mlp_spec %>% fit(Y~., data = df)
new_df <- data.frame(matrix(rnorm(1000), 100, 10))
colnames(new_df) <- paste("X", c(1:10), sep = "")
prediction1 = model0 %>% predict(new_data = data.frame(matrix(rnorm(1000), 100, 10)))
preds = prediction1$.pred

install.packages("tidytuesdayR")
library(tidytuesdayR)
coffee <- invisible(tidytuesdayR::tt_load(2020, week = 28)$coffee)

mlp_reg_spec <- mlp(hidden_units = 5, penalty = 0.1) %>%
set_mode('regression') %>% set_engine('nnet')

set.seed(345)
reg_split <- initial_split(mtcars, prop = 0.8)
reg_fit <- mlp_reg_spec %>% fit(mpg ~., data =training(reg_split))
reg_predictions <- reg_fit %>% predict(testing(reg_split))
print(head(reg_predictions))

reg_split <- initial_split(mtcars, prop = 0.8)
reg_fit <- mlp_reg_spec %>% fit(mpg ~., data = training(reg_split))
reg_predictions <- reg_fit %>% predict(testing(reg_split))
print(head(reg_predictions))

mlp_reg_spec <- mlp(hidden_units = 10, penalty = 0.25) %>%
set_mode('regression') %>% set_engine('nnet')


coffee <- invisibble(tidytuesdayR::tt_load(2020, week = 28)$coffee)
coffee_split <- initial_split(coffee, prop = 0.8)
coffee_train <- training(coffee_split)
coffee_test <- testing(coffee_split)
coffee_recipe <- recipe(coffee_train) %>%
update_role(everything(), new_role = 'support') %>%
#find the largest i such that the importance for 
#




reg_split <- initial_split(mtcars, prop = 0.75)
reg_fit <- mlp_reg_spec %>% fit(mpg ~. data = training(reg_split))
reg_predictions <- reg_fit %>% predict(testing(reg_split))
print(head(reg_predictions))

mlp_reg_spec <- mlp(hidden_units = 10, penalty = 0.25) %>%
set_mode('regression') %>% set_engine('nnet')






























#generate 1000 random counts with mean of 10 and size of 2.5
data <- rbinom(n = 1000, size = 2.5, mu = 10)
data <- rnbinom(n. =100)


#

rpois_inverse <- function(lambda, n = 1){
  result <- numeric(n)
  for(j in 1:n){
    u <- runif(1)
    k <- 0
    F_k <- exp(-lambda)
    while(u > F_k){
      k <- k + 1
      F_k <- F_k + dpois(k, lambda)
    }#finding that F_k so.that the ecdf is smallest than u -> traverse the quantile
    result[j] = k
  }
  return(result)
}


#SELECT A1, A2
#FROM GRP1 
#GROUP BY COL1, COL2
#ORDER BY COL2, COL1

rpois_man <- function(lambda, n = 1){
  replicate(n, {
  	time <- 0
  	count <- 0
  	while(time < 1){
  	  u <- runif(1)
  	  time <- time + (-log(u)/lambda)
  	  if(time < 1){
  	    count <- count + 1
  	  }
  	}
  	return(count)
  	})
}




pacman::p_load(car)
library(car)
model <- lm(mpg ~ disp + hp + wt + drat, data = mtcars)
vif(model)
which(vif(model) > 5)



#fetch the summary statistic in the coef(suumary(model)) here
#coef(summary(model)) --->


rowSd()
rowScale()#d is the degree here.
#coef(summary(model))
#model <- lm(mpg ~ disp + hp + wt + drat, data = mtcars)
#vif(model) 
#which(vif(model) > 5)


#
pacman::p_load(car)
library(car)
model <- lm(mpg ~ disp + hp + wt + drat, data = mtcars)
vif(model)
which(vif(model) > 5)
pacman::p_load(car)



beta_hat <- 0.5
se <- 0.12

simulated_data <- rnbinom(n = 1000, size = 2, mu = 10)
dnbinom(x = 5, size = 2.5, mu = 10)
pnbinom(x = 5)

#p

#\tilde{p} + pnrom()

beta_lower <- #
beta_higher <- #

lr_spec <- logistic_reg() %>%
set_engine("glm") %>%
set_mode('classification')



lr_spec 

ranger(Y~., 
	data = irs, num.trees = 500,
	importance = 'impurity')

regr_model <- ranger(Ozone ~., data = clean_airquality,
    num.trees = 500,
    importance = 'impurity')
predict(class_model)


#The bootstrap c.i. would be less reliable in terms of 
#omitting the tail distribution/quantile -> 
#will introduce severe under-covarage ->
#A nominal 95% CI might only cover the true 
#mean 80% of the time -> 
#The bootstrap or BcA -> 
#Fisher's exact test via hyper


rhyper(nn, m, n, k)
#Do some of the parametric testing procedure 
#as well as the regularized procedure -> beta-binomial(jeffrey's prior)
#is very helpful here.
#rhyper(nn, m, n, k)

#The bootstrap failure -> Var = n * K/N (1 - K/N) * (N-n)/(N-1)
#The bootstrap failure -> \hat{F}_{n} ---> F
#When n = 15 here.



d1 <- runif(1000)
poisson <- qpois(d1, lambda = 4)
hist(poisson)



X1 <- rnorm(1000)
X2 <- rnorm(1000)


df[df$country == 'US' & df$timestamp > 80, 'observed'] <- 
df[df$country == 'US' & df$timestamp > 80, 'pred_mean'] - 
runif(sum(df$country == 'US' & df$timestamp > 80), 0.8, 1.2)


df_apply <- apply(df, 1, function(X){
  (X - mean(X))/sd(X)
})


mutate(
    cusum_3 = rollapply(
        z, width = 3,
        FUN = sum, fill = NA,
        align = 'right', na.rm = FALSE
    	)
	) %>% ungroup() %>%
mutate(raw_alert = ifelse(!is.na(cusum_3) & cusum_3 < -3.8, TRUE, FALSE)) %>%
mutate(p_raw = 2 * pnorm(-abs(z))) #if z is NA, p_raw would be NA here.

%>% filter(impact_dau > 50000)
%>% arrange(desc(impact_dau))

%>% filter(impact_dau > 50000)
%>% arrange(desc(impact_dau))


mutate(cusum_3 = rollapply(
    z, width = 3,
    FUN = sum, fill = NA,
    align = 'right', na.rm = FALSE
)) %>% ungroup() %>% mutate()







result <- df %>% mutate(z = (observed - pred_mean)/pred_sd) %>%
group_by(country) %>%
arrange(timestamp, .by_group = TRUE)















# while p_accu < u -> 
#p_accu += dpois()






Attribution = case_when(
  satisfaction_score < 0.3 & years_at_company > 3 ~ "Yes",
  monthly_income < 3000 ~ "Yes",
  TRUE ~ sample(c("Yes", "No"), 1, prob = c(0.1, 0.9))
) %>% as.factor() %>%
mutate(
  satisfaction_score = if_else(runif(n) < 0.075, NA_real_, satisfaction_score),
  department = if_else(runif(n) < 0.035, NA_character_, department)
	)
  satisfaction_score < 0.3 & years_at_company > 3 ~ "Yes",
  monthly_income < 3000 ~ "Yes",
  TRUE ~ sample(c("Yes", "No"), 1, prob = c(0.1, 0.9)) %>%
  as.factor() %>%
  mutate(satisfaction_score = if_else(runif(n) < 0.075, NA_real_,
    satisfaction_score),
  ) %>% arrange(desc(monthly_income))

p_obs <- dhyper(x = mat[1, 1], m = col1, n = total - col1,
k = row1)
#k = row1, n = total - col1, m = col1
#x = mat[1, 1]
#p_more_extreme - p_obs
#p_mid <- p_strictly_greater + p_obs * 0.5

#p_obs ,


if_else(runif(n) < 0.075, NA_real_,
    satisfaction_score) %>% arrange(desc(monthly_income))

satisfaction_score < 0.3 & years_at_company > 3 ~ "Yes",
monthly_income < 3000 ~ "Yes",
TRUE ~ sample(c("Yes", "No"), 1, prob = c(0.1, 0.9))
%>% as.factor() %>% mutate()
####-> 
p_obs <- dhyper(x = mat[1, 1], m = col1, n = total - col1, 
k = row1)


set.seed(24)
N_true <- 1000
K <- 200
n <- 150
X <- rhyper(1, m = K, n = N_true - K, k = n)




p_obs <- dhyper(x = mat[1, 1], m = col1, n = total - col1,
  k = row1)
p_mid <- p_strictly_greater + p_obs * 0.5

qf(0.95, df1, df2) #(x_df/df1)
#qhyper(0.975, m, n, k) -> mk/n

prop.test(0, 5, correct = FALSE)$conf.int
p_hat <- 0
wald_ci <- p_hat + c(-1, 1) * qnorm(0.975) * sqrt(p_hat * (1 - p_hat)/5)
print(wald_ci)

prop.test(0, 5, correct = FALSE)$conf.int
p_hat <- 0
wald_ci <- p_hat + c(-1, 1) * qnorm(0.975) * sqrt(p_hat * (1 - p_hat)/5)

#give me a gamma-chisquare prior family 

qf(0.975, df1, df2)
prop.test(0, 5, correct = FALSE)$conf.int
print(wald_ci)
prop.test(0, 5, correct = FALSE)$conf.int
X <- rhyper(1, m = K, n = N_true - K, k = n)
K <- 200
n <- 150
N_true <- 1000
set.seed(2026)





my_qpois_concise <- function(U, lambda, max_k = 100){
  which.max(cumsum(dpois(0:max_k, lambda)) >= u) - 1
}

my_qpois_concise(0.8, 3)#finding the cumsum value here.
#X <- rhyper(1, m = K, n = N_true - K, k = n)
#prop.test(0, 5, correct = FALSE)$conf.int
#power analysis and quality control for that value here:

#X <- rhyper(1, m = K, n = N_true - K, k = n) here for the cumsum value here.

my_qpois_concise(0.8, 3)


set.seed(234)
u_samples <- runif(10000)
concise_samples <- sapply(u_samples, my_qpois_concise, lambda = 3)


#m = a, n = N- a, k =b -> searching for the likelihood 
#E[c] = k * m/(m+n) here

possibleN <- 100:10000
likelihood <- numeric(length(possibleN))
a <- 50
b <- 60
observed_c <- 12 
for(i in seq_along(possibleN)){
  N <- possibleN[i]
  sims <- rhyper(10000, m = a, n = N - a, k = b)
  likelihood[i] = mean(sims == observed_c)
}
best_N <- possibleN[which.max(likelihood)]

#Help you find that specific indices for that procedure.

#P(X_{k+1} = 1 | x vacant in the first k ) = (M - x)/(N - k)
#(M - x)/(N - k) exactly here!

#P(X_{k+1} = 1|) p
#develop some recursive procedures here:
#E[D_{T_{m}}] -> from m = 0,1,2,...,N -> pick the m* that
#gives the minimum value here:

rowSum(X1) 
#
#Finding the column specific max with the row specific mean here!
X1[apply(X1, 2, function(X){which.max(X)}), apply(X1, 1, function(X){which.min(X)})]


#P(X_{k+1} = 1) -> (M - x)/(N - k) right?

for(i in seq_along(possibleN)){
  N <- possibleN[i]
  sims <- rhyper(10000, m = a, n = N - a, k = b)
  likelihood[i] <- mean(sims == observed_c)
}
best_N <- possibleN[which.max(likelihood)]


library(tidyr)

# Step 1: Aggregate by Year-Month
monthly_agg <- sales_df %>%
  group_by(year, month) %>%
  summarise(total_revenue = sum(revenue, na.rm = TRUE), .groups = 'drop')

# Step 2: Pivot Wider (to compare months as columns)
monthly_wide <- monthly_agg %>%
  pivot_wider(
    names_from = month,        # Month numbers become column names
    values_from = total_revenue # Fill values with revenue
  )
# Result: Columns -> year, 1, 2, 3, ... 12

# Step 3: Pivot Longer (reverse - for plotting)
monthly_long <- monthly_wide %>%
  pivot_longer(
    cols = -year,               # Keep year, pivot all month columns
    names_to = "month",
    values_to = "revenue"
  )



#if f(x)/g(x) supremum -> 

#What happens if f(x)/g(x) get the supreme


monthly_long <- monthly_wide %>% 
pivot_longer(
    cols = -yeaer,
    names_to = 'month',
    values_to = 'revenue'
	)



X1 = np.random.permutation()

grid_list <- seq(-10000, 10000, 0.01)
M_sup <- 


X_explain = np.random.random((1, 50, 8))
X_train = np.random.random((100, 50, 8))
def model_predict(X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().reshape(-1, 1)
background = X_train[:100]  

explainer = timeshap.Explainer(
    model=model_predict,
    background=background,
    return_hidden_state=False  
)

explainer = timeshap.Explainer(
    model = model_predict,
    background = background,
    return_hidden_state = False
)
event_explanation = explainer.local_event(X_explain)
feature_explanation = explainer.local_feature(X_explain)  
cell_explanation = explainer.local_cell(X_explain)
#Getting the instance value explanations for the visualizations:
instances = X_explain 
global_event = explainer.global_event(instances)
global_feature = explainer.global_feature(instances) 
event_shap_values, event_names = event_explanation
feature_shap_values, feature_names = feature_explanation


















































































































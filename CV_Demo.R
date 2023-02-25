
# Setup ####
d = mtcars
insample_fraction  = .9
seed = 1
set.seed(seed)
target_name = 'mpg'
preds_names = setdiff(names(mtcars),target_name)


# Train N Test ####
n = dim(d)[1]
idx_iss = sample(n,n*insample_fraction)
idx_oos = setdiff(1:n,idx_iss)
X_iss = as.matrix(d[idx_iss,preds_names])
y_iss = d[idx_iss,target_name]
X_oos = as.matrix(d[idx_oos,preds_names])
y_oos = d[idx_oos,target_name]
p = dim(X_iss)[2]


# Matrices ####
MyMatrixMaker = function(X){
  X_std = scale(x = X, center = TRUE, scale = FALSE)
  out = as.matrix(X_std)
  return(out)
}
X_iis = MyMatrixMaker(X_iss)
y_iis = MyMatrixMaker(y_iss)
X_oos = MyMatrixMaker(X_oos)
y_oos = MyMatrixMaker(y_oos)

# CV Setup ####

lambdas <- 10^seq(-3, 5, length.out = 100)

nfolds = 4
n_cv = dim(X_iss)[1]
n_oos_cv = floor(n_cv/nfolds)
n_iss_cv = n_cv - n_oos_cv

idx_randsort = sample(1:n_cv,n_cv)

n_lambdas = length(lambdas)

MSE_mat = matrix(NA,nfolds,n_lambdas)


# CV ####

idx_range = 1:n_oos_cv

for (i_fold in 1:nfolds){
  
  idx_oos_cv = idx_randsort[idx_range+n_oos_cv*(i_fold-1)]
  idx_iss_cv = setdiff(idx_randsort,idx_oos_cv)
  
  X_iis_cv = X_iis[idx_iss_cv,]
  y_iis_cv = y_iis[idx_iss_cv]  
  X_oos_cv = X_iis[idx_oos_cv,]
  y_oos_cv = y_iis[idx_oos_cv]
  
  for (i_lambda in 1:n_lambdas){
    fit_i = glmnet::glmnet(x = X_iis_cv,
                           y = y_iis_cv,
                           family = 'gaussian',
                           lambda = lambdas[i_lambda])
    
    yhat_i = predict(fit_i, newx = X_oos_cv)
    
    MSE_i = sum((yhat_i-y_oos_cv)^2)/n_oos_cv
    
    MSE_mat[i_fold,i_lambda] = MSE_i
    
  }
}

# Plot ####


plot(x = log(lambdas),
     y= rep(0,n_lambdas),
     ylim = c(0,max(MSE_mat)), ylab = 'MSE',
     cex=.001)

for (i in 1:nfolds){
  points(x = log(lambdas), y = MSE_mat[i,],col = i+1)
}


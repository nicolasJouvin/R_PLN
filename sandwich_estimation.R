library(torch)
library(R6)
#setwd("~/Documents/These/R_PLN")
source(file = 'utils.r')
source(file = 'IMPS_PLN.r')
source(file = 'VEM.r')


d = 1L
n = 2000L
p = 2000L
q = 10L




### Sampling some data ###
O <-  torch_tensor(matrix(0,nrow = n, ncol = p))
covariates <- torch_tensor(matrix(rnorm(n*d),nrow = n, ncol = d))
true_Theta <- torch_tensor(matrix(rnorm(d*p),nrow = d, ncol = p))
true_C <- torch_tensor(matrix(rnorm(p*q), nrow = p, ncol = q) )/3
true_Sigma <- torch_matmul(true_C,torch_transpose(true_C, 2,1))
true_Theta <- torch_tensor(matrix(rnorm(d*p),nrow = d, ncol = p))/2
true_C <- C_from_Sigma(true_Sigma,q)
Y <- sample_PLN(true_C,true_Theta,O,covariates)
n_a = 200
n_b = 300
vizmat(as.matrix(true_Sigma[n_a:n_b, n_a:n_b]))
nb_iter = 15
var_percentage = torch_zeros(nb_iter)
imps_percentage = torch_zeros(nb_iter)

for ( i in 1:nb_iter){
  Y <- sample_PLN(true_C,true_Theta,O,covariates)
  pln = VEM_PLN$new(Y,O,covariates)
  pln$fit(2,0.1, verbose = TRUE)
  inv_Fischer = pln$get_variational_inv_Fisher()
  res_var = how_much(pln$Theta, true_Theta, inv_Fischer, p*d-5L,Y$shape[1])
  var_percentage[i] = res_var 
  pr('res_var', res_var)
  imps <- IMPS_PLN$new(Y,O,covariates,q)
  imps$C <- torch_clone(true_C) #torch_clone(true_C) + 0*torch_randn(true_C$shape)
  imps$Theta <- pln$Theta#true_Theta + 0.4*torch_randn(true_Theta$shape)
  imps$fit(1L, acc = 0.008, lr = 0)
  hess_log_p_theta <- imps$compute_hess_log_p_theta()
  res_imps = how_much(imps$Theta, true_Theta, -hess_log_p_theta, p*d - 5L, Y$shape[1]) 
  pr('res_imps', res_imps)
  imps_percentage[i] = res_imps
}

pr('mean var', mean_var)
pr('mean imps', mean_imps/nb_iter)

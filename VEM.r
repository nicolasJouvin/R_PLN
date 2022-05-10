library(torch)
library(R6)
source(file = 'utils.r')




Y <- torch_tensor(as.matrix(read.csv('Y_test')))
O <- torch_tensor(as.matrix(read.csv('O_test')))
covariates <- torch_tensor(as.matrix(read.csv('cov_test')))
true_Sigma <- torch_tensor(as.matrix(read.csv('true_Sigma_test')))
true_C <- torch_cholesky(true_Sigma)
true_B_zero <- torch_tensor(as.matrix(read.csv('true_beta_test')))

#Variational parameters 


first_closed_Sigma<- function(M,S){
  #closed form for Sigma 
  n = M$shape[1]
  return(1/n*(torch_matmul(torch_transpose(M,2,1),M) + torch_diag(torch_sum(torch_multiply(S,S), dim = 1)))) 
}
ELBO <- function(Y, O, covariates, M, S, Sigma, B_zero){
  n = Y$shape[1]
  p = Y$shape[2]
  inv_Sigma <- torch_inverse(Sigma)
  Gram_matrix <- torch_matmul(covariates,B_zero) 
  help_calculus <- O + Gram_matrix + M 
  SrondS = torch_multiply(S,S)
  tmp = -n/2*torch_logdet(Sigma) 
  tmp = torch_add(tmp,torch_sum(-torch_exp(help_calculus+ SrondS/2) + torch_multiply(Y, help_calculus) + 1/2*torch_log(SrondS)))
  tmp = torch_sub(tmp,1/2*torch_trace(torch_matmul(torch_matmul(torch_transpose(M,2,1), M) + torch_diag(torch_sum(SrondS, dim = 1)), inv_Sigma)))
  tmp = torch_sub(tmp, torch_sum(log_stirling(Y)))
  tmp = torch_add(tmp, n*p/2)
  return(tmp)
}



VEM_PLN <- R6Class("VEM_PLN", 
                   public = list(
                     Y = NULL, 
                     O = NULL,
                     covariates = NULL, 
                     p = NULL, 
                     n = NULL, 
                     d = NULL, 
                     M = NULL, 
                     S = NULL, 
                     Sigma = NULL, 
                     B_zero = NULL,
                     initialize = function(Y,O,covariates){
                       self$Y <- Y
                       self$O <- O
                       self$covariates <- covariates
                       self$p <- Y$shape[2]
                       self$n <- Y$shape[1]
                       self$d <- covariates$shape[2]
                       ## Variational parameters
                       self$M <- torch_zeros(self$n, self$p, requires_grad = TRUE)
                       self$S <- torch_ones(self$n, self$p, requires_grad = TRUE)
                       ## Model parameters 
                       self$B_zero <- torch_zeros(self$d,self$p, requires_grad = TRUE)
                       self$Sigma <- torch_eye(self$p)
                     },
                     
                     fit = function(N_iter, lr){
                       optimizer = optim_rprop(c(self$B_zero, self$M, self$S), lr = lr)
                       for (i in 1:N_iter){
                         optimizer$zero_grad()
                         loss = - ELBO(self$Y, self$O, self$covariates, self$M, self$S, self$Sigma, self$B_zero)
                         #pr('ELBO', (-loss$item())/(self$n))
                         loss$backward()
                         optimizer$step()
                         self$Sigma <- first_closed_Sigma(self$M, self$S)
                         pr('ELBO', -loss$item())
                         pr('MSE', MSE(self$Sigma - true_Sigma))
                       }
                       
                     }
                   )
                   )



pln = VEM_PLN$new(Y,O,covariates)
pln$fit(100,0.1)
pr('MSE', MSE(pln$Sigma - true_Sigma))

print(torch_sum(log_stirling(Y)))
vizmat(as.matrix(pln$Sigma))
print(pln$B_zero)
print(true_B_zero)






pln = VEM_PLN$new(Y,O,covariates)
pln$fit(600,0.1)
pr('MSE', MSE(pln$Sigma - true_Sigma))




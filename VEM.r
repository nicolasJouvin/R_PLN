library(torch)
library(R6)
source(file = 'utils.r')




Y <- torch_tensor(as.matrix(read.csv('Y.csv')))
O <- torch_tensor(as.matrix(read.csv('O.csv')))
covariates <- torch_tensor(as.matrix(read.csv('covariates.csv')))
true_Sigma <- torch_tensor(as.matrix(read.csv('true_5_Sigma.csv')))
#true_C <- torch_cholesky(true_Sigma)
true_Theta <- torch_tensor(as.matrix(read.csv('true_beta.csv')))

#Variational parameters 


first_closed_Sigma<- function(M,S){
  #closed form for Sigma 
  n = M$shape[1]
  return(1/n*(torch_matmul(torch_transpose(M,2,1),M) + torch_diag(torch_sum(torch_multiply(S,S), dim = 1)))) 
}
ELBO <- function(Y, O, covariates, M, S, Sigma, Theta){
  n = Y$shape[1]
  p = Y$shape[2]
  inv_Sigma <- torch_inverse(Sigma)
  Gram_matrix <- torch_matmul(covariates,Theta) 
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
                     A = NULL, 
                     Sigma = NULL, 
                     Theta = NULL,
                     initialize = function(Y,O,covariates){
                       self$Y <- Y
                       self$O <- O
                       self$covariates <- covariates
                       pr('covariates', self$covariates)
                       self$p <- Y$shape[2]
                       self$n <- Y$shape[1]
                       self$d <- covariates$shape[2]
                       ## Variational parameters
                       self$M <- torch_zeros(self$n, self$p, requires_grad = TRUE)
                       self$S <- torch_ones(self$n, self$p, requires_grad = TRUE)
                       ## Model parameters 
                       self$Theta <- torch_zeros(self$d,self$p, requires_grad = TRUE)
                       self$Sigma <- torch_eye(self$p)
                     },
                     
                     fit = function(N_iter, lr){
                       optimizer = optim_rprop(c(self$Theta, self$M, self$S), lr = lr)
                       for (i in 1:N_iter){
                         optimizer$zero_grad()
                         loss = - ELBO(self$Y, self$O, self$covariates, self$M, self$S, self$Sigma, self$Theta)
                         loss$backward()
                         optimizer$step()
                         self$Sigma <- first_closed_Sigma(self$M, self$S)
                         pr('ELBO', -loss$item()/(self$n))
                         pr('MSE Sigma', MSE(self$Sigma - true_Sigma))
                         pr('MSE Theta', MSE(self$Theta- true_Theta))
                       }
                     }, 
                     
                     get_Dn = function(){
                       self$A = torch_exp(self$O + torch_matmul(self$covariates,self$Theta) + self$M + 1/2*(self$S)**2)
                       YmoinsA = self$Y - self$A 
                       outer_prod_YmoinsA = torch_matmul(YmoinsA$unsqueeze(3), YmoinsA$unsqueeze(2))
                       outer_prod_X = torch_matmul(self$covariates$unsqueeze(3), self$covariates$unsqueeze(2))
                       pr('outer prod shape', outer_prod_YmoinsA$shape)
                       pr('outer_prod X shape', outer_prod_X$shape)
                       kron 
                       pr('res', torch_kron(outer_prod_YmoinsA, outer_prod_X))
                       }
                   )
                   )



pln = VEM_PLN$new(Y,O,covariates)
pln$fit(1,0.1)
pln$get_Dn()
#pr('MSE', MSE(pln$Sigma - true_Sigma))
x = torch_randn(100,10,10)
y = torch_randn(1,10,10)
torch_kron(x,y)

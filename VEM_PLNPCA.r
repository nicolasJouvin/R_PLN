library(torch)
library(R6)
source(file = '../R_PLN/utils.r')
# 
# 
# Y <- torch_tensor(as.matrix(read.csv('Y.csv')))
# O <- torch_tensor(as.matrix(read.csv('O.csv')))
# covariates <- torch_tensor(as.matrix(read.csv('covariates.csv')))
# true_Sigma <- torch_tensor(as.matrix(read.csv('true_5_Sigma.csv')))
# #true_C <- torch_cholesky(true_Sigma)
# true_Theta <- torch_tensor(as.matrix(read.csv('true_beta.csv')))



ELBO_PCA <- function(Y, O, covariates, M, S, C, Theta){
  ## compute the ELBO with a PCA parametrization'''
  n = Y$shape[1]
  q = C$shape[2]
  # Store some variables that will need to be computed twice
  A = O + torch_mm(covariates, Theta) + torch_mm(M, C$t())
  SrondS = torch_multiply(S, S)
  # Next, we add the four terms of the ELBO_PCA (plus the useless factorial)
  YA = torch_sum(torch_multiply(Y, A))
  moinsexpAplusSrondSCCT = torch_sum(-torch_exp(A + 1 / 2 *
                                                  torch_mm(SrondS, torch_multiply(C, C)$t())))
  moinslogSrondS = 1 / 2 * torch_sum(torch_log(SrondS))
  MMplusSrondS = torch_sum(-1 / 2 * (torch_multiply(M, M) + torch_multiply(S, S)))
  log_stirlingY = torch_sum(log_stirling(Y))
return(YA + moinsexpAplusSrondSCCT + moinslogSrondS + MMplusSrondS - log_stirlingY + n * q / 2)
}

VEM_PLNPCA <- R6Class("VEM_PLNPCA", 
                   public = list(
                     Y = NULL, 
                     O = NULL,
                     covariates = NULL, 
                     p = NULL, 
                     q = NULL,
                     n = NULL, 
                     d = NULL, 
                     M = NULL, 
                     S = NULL, 
                     A = NULL, 
                     C = NULL,
                     Sigma = NULL, 
                     Theta = NULL,
                     good_init = NULL,
                     fitted = NULL, 
                     ELBO_list = NULL, 
                     initialize = function(Y,O,covariates,q, good_init = TRUE){
                       self$Y <- Y
                       self$O <- O
                       self$covariates <- covariates
                       self$q = q
                       self$good_init = good_init
                       self$p <- Y$shape[2]
                       self$n <- Y$shape[1]
                       self$d <- covariates$shape[2]
                       ## Variational parameters
                       self$M <- torch_zeros(self$n, self$q, requires_grad = TRUE)
                       self$S <- torch_ones(self$n, self$q, requires_grad = TRUE)
                       ## Model parameters 
                       message('Initialization ...')
                       if (self$good_init){
                          self$Theta <- Poisson_reg(Y,O,covariates)$detach()$clone()$requires_grad_(TRUE)
                          self$C <- init_C(Y,O,covariates, self$Theta,q)$detach()$clone()$requires_grad_(TRUE)
                       }
                       else{
                         self$Theta <- torch_zeros(self$d, self$p, requires_grad = TRUE)
                         self$C <- torch_randn(self$p, self$q, requires_grad = TRUE)
                       }
                       message('Initialization finished.')
                       self$fitted = FALSE
                       self$ELBO_list = c()
                     },
                     get_Sigma = function(){
                       return(torch_mm(self$C, self$C$t()))
                     },
                     fit = function(N_iter, lr, verbose = FALSE){
                       optimizer = optim_rprop(c(self$Theta, self$C, self$M, self$S), lr = lr)
                       for (i in 1:N_iter){
                         optimizer$zero_grad()
                         loss = - ELBO_PCA(self$Y, self$O, self$covariates, self$M, self$S, self$C, self$Theta)
                         loss$backward()
                         optimizer$step()
                         if(verbose && (i%%50 == 0)){
                           message('i :', as.character(i) )
                           message('norm sig', as.character(torch_norm(self$get_Sigma())$item()))
                           message('ELBO : ', as.character(-loss$item()/(self$n)))
                           pr('MSE Sigma', MSE(self$get_Sigma() - true_Sigma))
                           pr('MSE Theta', MSE(self$Theta- true_Theta))
                         }
                         self$ELBO_list = c(self$ELBO_list, -loss$item()/(self$n))
                       }
                      self$fitted <- TRUE 
                     }, 
                  plot_log_neg_ELBO = function(from = 10){
                    plot(log(-self$ELBO_list[from:length(self$ELBO_list) ]))
                  },
                  getLatentVariables = function(take_cov = FALSE){
                        if (take_cov){
                            return( torch_matmul(self$M$unsqueeze(2),self$C$t()$unsqueeze(1))$squeeze()+  torch_mm(self$covariates, self$Theta))
                           }
                        else{
                             return(torch_matmul(self$M$unsqueeze(2),self$C$t()$unsqueeze(1))$squeeze())
                             }
                    },
                    getPostSigma = function(){
                        return(torch_matmul(torch_matmul(self$C, 1/self$n*(torch_matmul(self$M$t(),self$M) +torch_diag(torch_sum(torch_multiply(self$S,self$S), dim = 1)))), self$C$t()))
                       }
                   )
)

plnpca = VEM_PLNPCA$new(Y,O,covariates, q, good_init = FALSE)
plnpca$fit(30,0.1,verbose = TRUE)
plnpca$plot_log_neg_ELBO()

# plnpca$Theta
# true_Theta
# covariates
# 
# d = 1L
# n = 2000L
# p = 2000L
# q = 10L




### Sampling some data ###
# O <-  torch_tensor(matrix(0,nrow = n, ncol = p))
# covariates <- torch_tensor(matrix(rnorm(n*d),nrow = n, ncol = d))
# true_Theta <- torch_tensor(matrix(rnorm(d*p),nrow = d, ncol = p))
# true_C <- torch_tensor(matrix(rnorm(p*q), nrow = p, ncol = q) )/3
# true_Sigma <- torch_matmul(true_C,torch_transpose(true_C, 2,1))
# true_Theta <- torch_tensor(matrix(rnorm(d*p),nrow = d, ncol = p))/2
# true_C <- C_from_Sigma(true_Sigma,q)
# Y <- sample_PLN(true_C,true_Theta,O,covariates)
# n_a = 200
# n_b = 300
# vizmat(as.matrix(true_Sigma[n_a:n_b, n_a:n_b]))




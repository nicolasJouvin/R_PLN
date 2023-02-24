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

fastELBO <-  function(Y, O, covariates, M, S, Sigma, beta){
    ###Compute the ELBO (Evidence LOwer Bound. See the doc for more details
    #on the computation.

    #Args:
    #    Y: torch.tensor. Counts with size (n,p)
    #    0: torch.tensor. Offset, size (n,p)
    #    covariates: torch.tensor. Covariates, size (n,d)
    #    M: torch.tensor. Variational parameter with size (n,p)
    #    S: torch.tensor. Variational parameter with size (n,p)
    #    Sigma: torch.tensor. Model parameter with size (p,p)
    #    beta: torch.tensor. Model parameter with size (d,p)
    #Returns:
    #    torch.tensor of size 1 with a gradient. The ELBO.
    ###
    n <- Y$shape[1] 
    p <- Y$shape[2]
    SrondS <- torch_multiply(S, S)
    OplusM <- O + M
    MmoinsXB <- M - torch_mm(covariates, beta)
    
    tmp <- - n / 2 * torch_logdet(Sigma)
    tmp = tmp + torch_sum(torch_multiply(Y, OplusM)
                    - torch_exp(OplusM + SrondS / 2)
                    + 1 / 2 * torch_log(SrondS)
                    )
    DplusMmoinsXB2 = torch_diag(
        torch_sum(SrondS, dim=1)) + torch_mm(MmoinsXB$t(), MmoinsXB)
    tmp = tmp -1 / 2 * torch_trace(
        torch_mm(
            torch_inverse(Sigma),
            DplusMmoinsXB2
        )
    )
    tmp = tmp-torch_sum(log_stirling(Y))
    tmp = tmp + n * p / 2 
    return(tmp)
}

fastClosedSigma <- function(covariates, M, S, beta){
        ### Closed form for Sigma for the M step.
        n = M$shape[1]
        p = M$shape[2]
        MmoinsXB = M - torch_mm(covariates, beta)
        closed = torch_mm(MmoinsXB$t(), MmoinsXB)
        closed = closed + torch_diag(torch_sum(torch_multiply(S, S), dim=1))
        return (1 / (n) * closed)
    }

fastClosedTheta<- function(covariates, M){
    ##Closed form for beta for the M step
    return(torch_mm(torch_mm(torch_inverse(torch_mm(covariates$t(),covariates)),covariates$t()),M))
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
                     ELBO_list = NULL, 
                     goodInit = NULL, 
                     fast = NULL,
                     initialize = function(Y,O,covariates, goodInit = TRUE, fast = TRUE){
                       self$Y <- Y
                       self$O <- O
                       self$covariates <- covariates
                       self$p <- Y$shape[2]
                       self$n <- Y$shape[1]
                       self$d <- covariates$shape[2]
                       self$goodInit = goodInit
                       self$fast = fast
                       print('Initialization...')
                       ## Model parameters 
                       if (self$goodInit) {
                           self$Theta <- Poisson_reg(Y,O,covariates)$detach()$clone()$requires_grad_(TRUE)
                           self$Sigma <- init_Sigma(self$Y, self$O, self$covariates,self$Theta)$detach()$clone()
                       }
                       else{
                            self$Theta <- torch_zeros(self$d,self$p, requires_grad = TRUE)
                            self$Sigma <- torch_eye(self$p)
                       }
                       ## Variational parameters
                       self$M <- torch_zeros(self$n, self$p, requires_grad = TRUE)
                       self$S <- torch_ones(self$n, self$p, requires_grad = TRUE)
                       print('Initialization finished.')
                       self$ELBO_list = c()
                     },
                     
                     fit = function(N_iter, lr, verbose = FALSE){
                       if (self$fast) {
                           optimizer = optim_rprop(c(self$M, self$S), lr = lr)
                       }
                       else{
                           optimizer = optim_rprop(c(self$Theta, self$M, self$S), lr = lr)
                       }
                       for (i in 1:N_iter){
                         optimizer$zero_grad()
                         
                         if(self$fast){
                            loss = - fastELBO(self$Y, self$O, self$covariates, self$M, self$S, self$Sigma, self$Theta)
                            loss$backward()
                            optimizer$step()
                            self$Theta <- fastClosedTheta(self$covariates, self$M)
                            self$Sigma <- fastClosedSigma(self$covariates, self$M, self$S, self$Theta)
                             }
                         else{
                             self$A = torch_exp(self$O + torch_matmul(self$covariates,self$Theta) + self$M + 1/2*(self$S)**2)
                             loss = - ELBO(self$Y, self$O, self$covariates, self$M, self$S, self$Sigma, self$Theta)
                             loss$backward()
                             optimizer$step()
                             self$Sigma <- first_closed_Sigma(self$M, self$S)
                         }
                         
                         if(verbose && (i %% 50 == 0)){
                           message('i : ', as.character(i))
                           message('ELBO : ', as.character(-loss$item()/(self$n)))
                           #pr('MSE Sigma', MSE(self$Sigma - true_Sigma))
                           #pr('MSE Theta', MSE(self$Theta- true_Theta))
                         }
                         self$ELBO_list = c(self$ELBO_list, -loss$item()/(self$n))
                         
                       }
                     }, 
                     plot_log_neg_ELBO = function(from = 10){
                       plot(log(-self$ELBO_list[from:length(self$ELBO_list) ]))
                     }, 
                     
                     get_Dn_Theta = function(){
                       YmoinsA = self$Y - self$A 
                       outer_prod_YmoinsA = torch_matmul(YmoinsA$unsqueeze(3), YmoinsA$unsqueeze(2))
                       outer_prod_X = torch_matmul(self$covariates$unsqueeze(3), self$covariates$unsqueeze(2))
                       size_kron = outer_prod_YmoinsA$shape[2]*outer_prod_X$shape[2]
                       res = torch_zeros(size_kron, size_kron)
                       for ( i in 1:self$n){
                         res = res + torch_kron(outer_prod_YmoinsA[i,,],outer_prod_X[i,,])
                       }
                       return(res/self$n)
                       },

                    get_mat_i_Theta = function(i){
                      D_i = torch_diag(self$A[i,])
                      C_i = torch_diag(2/(2+self$S[i,]^4*self$A[i,]))
                      C_i_inv =torch_diag((2+self$S[i,]^4*self$A[i,])/2)
                      D_i_inv_sqrt = torch_diag(self$A[i,]**(-1/2))
                      D_i_sqrt = torch_diag(self$A[i,]**(1/2))
                      invB_i = C_i + torch_matmul(torch_matmul(D_i_inv_sqrt, self$Sigma), D_i_inv_sqrt)
                      B_i = torch_inverse(invB_i)
                      Big_mat = torch_mm(torch_mm(torch_mm(torch_mm(D_i_sqrt, C_i), C_i_inv -B_i),C_i), D_i_sqrt)  
                      return(Big_mat)
                      },
                    get_Cn_Theta = function(){
                      C_n = torch_zeros(self$d*self$p, self$d*self$p)
                      for (i in 1:(self$n)){
                        big_mat = self$get_mat_i_Theta(i)
                        x_i <-  self$covariates[i,]
                        xxt = torch_matmul(x_i$unsqueeze(2), x_i$unsqueeze(1))
                        C_n = C_n + torch_kron(big_mat, xxt)
                      }
                      return(-C_n/(self$n))
                    },
                    get_variational_inv_Fisher = function(){
                      Dn = self$get_Dn_Theta()
                      Cn = self$get_Cn_Theta()
                      inv_Dn = torch_inverse(Dn)
                      return(torch_mm(torch_mm(Cn, inv_Dn), Cn))
                    },
                    getLatentVariables = function(take_cov = FALSE){
                       if (self$fast){
                           if (take_cov){
                               return(self$M)
                               }
                           else{
                               return(self$M - torch_mm(self$covariates, self$Theta))
                               }
                           }
                       else{
                           if (take_cov){
                               return(self$M + torch_mm(self$covariates, self$Theta))
                               }
                           else{
                               return(self$M)
                               }
                           }
                        }

                   )
                   )


ELBO_PCA <- function(Y, O, covariates, M, S, C, Theta){
  ## compute the ELBO with a PCA parametrization'''
  n = Y$shape[1]
  q = C$shape[2]
  # Store some variables that will need to be computed twice
  A = O + torch_mm(covariates, Theta) + torch_mm(M, C$t())
  SrondS = torch_multiply(S, S)
  # Next, we add the four terms of the ELBO_PCA
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
                         #self$A = torch_exp(self$O + torch_matmul(self$covariates,self$Theta) + self$M + 1/2*(self$S)**2)
                         loss = - ELBO_PCA(self$Y, self$O, self$covariates, self$M, self$S, self$C, self$Theta)
                         loss$backward()
                         optimizer$step()
                         #self$Sigma <- first_closed_Sigma(self$M, self$S)
                         if(verbose && (i%%50 == 0)){
                           message('i :', as.character(i) )
                           message('norm sig', as.character(torch_norm(self$get_Sigma())$item()))
                           message('ELBO : ', as.character(-loss$item()/(self$n)))
                           #pr('MSE Sigma', MSE(self$get_Sigma() - true_Sigma))
                           #pr('MSE Theta', MSE(self$Theta- true_Theta))
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
                       #pr('first', torch_matmul(self$M$t(),self$M) +torch_diag(torch_sum(torch_multiply(self$S,self$S), dim = 1)))
                       #pr('sec', 
                        return(torch_matmul(torch_matmul(self$C, 1/self$n*(torch_matmul(self$M$t(),self$M) +torch_diag(torch_sum(torch_multiply(self$S,self$S), dim = 1)))), self$C$t()))
                       }
                   )
)

#plnpca = VEM_PLNPCA$new(Y,O,covariates, q, good_init = FALSE)
#plnpca$fit(30,0.1,verbose = TRUE)
#plnpca$plot_log_neg_ELBO()

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

ELBO_list  = list()
ELBO_list = c(ELBO_list, 1)
ELBO_list = c(ELBO_list, 1)
ELBO_list = c(ELBO_list, 1)
ELBO_list



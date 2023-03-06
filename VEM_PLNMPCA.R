library(torch)
library(R6)
library(aricode)
source(file = 'utils.r')

.clip_proba = function(x, zero = .Machine$double.eps) {
  x[x$isnan()]    <- zero
  x[x > 1 - zero] <- 1 - zero
  x[x <     zero] <-     zero
  x
}

log1pexp = function(x) {
  return(torch_log1p(torch_exp(-torch_abs(x))) + torch_relu(x))
}

logexpm1= function(x) {
  return(x + torch_log(-torch_expm1(-x)))
}


reals_to_spd_matrix <- function(x, scale=torch_tensor(1.0)) {
  #' Parameterization of a positive definite matrix by a vector
  #' Hijacked from Jean-Benoist Léger's parameterization cookbook 
  #' and Python code : https://gitlab.com/jbleger/parametrization-cookbook/-/blob/main/parametrization_cookbook/functions/torch.py#L335
  #' Convention is 
  #'  * diagonal values are in x[:n] 
  #'  * lower triangular values are un x[n:]
  device = x$device
  n = as.integer((8 * x$shape + 1) ^ 0.5 / 2)
  stopifnot(
    "x must be a 1-D tensor" = length(x$shape) == 1
  )
  stopifnot(
    "Incorect size. It does not exist n such as n*(n+1)/2=={x.shape[-1]}" = x$shape == as.integer(n * (n + 1) / 2)
  ) 
  
  stopifnot(
    "Non broacastable shapes, got matrix shape {x$shape} and scale shape {scale$shape}" = 
      is.numeric(scale) | (length(scale$shape) == 1 & (scale$shape[0] == n | scale$shape[0] == 1))
    )

  y = torch_zeros(c(n, n), device=device, dtype=x$dtype)
  diag_idx = matrix(as.logical(diag(n)), n,n)
  y[diag_idx] = log1pexp(x[1:n])
  
  lower_tri_idxs = lower.tri(y, diag = FALSE)
  y[lower_tri_idxs] = x[(n+1):N]
  y = y$divide(torch_sqrt(torch_arange(1, n, device=device))[,NULL])
  z_rescaled = y$matmul(y$t())
  
  if (length(scale$shape) == 1) {
    sqrt_scale = torch_sqrt(scale)
    z = z_rescaled$mul(sqrt_scale[, NULL])$mul(sqrt_scale[, NULL])
  } else if(is.numeric(scale)) {
    z = z_rescaled * scale
  }
  return(z)
}

sdp_matrix_to_reals = function(z, scale=torch_tensor(1.0)) {
  device = z$device
  stopifnot(
    "input z must be a square 2-D tensor (a matrix)" = length(z$shape) == 2 | z$shape[-2] != z$shape[-1]
  )
  stopifnot(
    "Non broacastable shapes, got matrix shape {x$shape} and scale shape {scale$shape}" = 
      is.numeric(scale) | (length(scale$shape) == 1 & (scale$shape[0] == n | scale$shape[0] == 1))
  )
  n = z$shape[-1]
  
  if (length(scale$shape) == 1) {
    sqrt_scale = torch_sqrt(scale)
    z_rescaled = z$div(sqrt_scale[,NULL])$div(sqrt_scale[NULL,])
  } else if(is.numeric(scale)) {
    z_rescaled = z / scale
  }
  
  y = linalg_cholesky(z_rescaled)
  y = y$mul(torch_sqrt(torch_arange(1, n, device=device))[,NULL])
  diag_values = y$diag()
  tril_values = y[lower.tri(y, diag = FALSE)]
  return(torch_cat(c(logexpm1(diag_values), tril_values)))
}

n=5
x = torch_arange(1, n*(n+1) / 2)
scale = torch_tensor(1e0)
print(x$shape)
z = reals_to_spd_matrix(x, scale = scale)
x_bis = sdp_matrix_to_reals(z, scale = scale)
torch_allclose(x, x_bis)

ELBO_MPCA <- function(Y, O, covariates, M, S, Tau, C, Theta, Lambda_vec, Mu, Pi){
  ## compute the ELBO with a PLN mixture of Commmon PCA parametrization
  # i.e. loadings are shared accross clusters and the scores follow a GMM
  # C_k = C \forall k
  
  n = Y$shape[1]
  q = C$shape[2]
  K = Tau$shape[1]
  
  # Transform Lambda_vec to sdp matrices with inversible transform
  Lambda = lapply(1:K, \(k) reals_to_spd_matrix(Lambda_vec[k,]$squeeze())[NULL,,]) %>% torch_cat
  
  # log p(Y | W) (obs. Poisson part)
  A = O + torch_mm(covariates, Theta) + torch_mm(M, C$t())
  SrondS = torch_multiply(S, S)
  YA = torch_sum(torch_multiply(Y, A))
  moinsexpAplusSrondSCCT = torch_sum(-torch_exp(A + 1 / 2 *
                                                  torch_mm(SrondS, torch_multiply(C, C)$t())))
  elbo = YA + moinsexpAplusSrondSCCT- torch_sum(log_stirling(Y))
  
  # log p(Z) (latent multinomial part)
  elbo = elbo + torch_sum(Tau * torch_log(Pi)[, NULL])
  
  # Entropy(q(W))
  logSrondS = (1 / 2) * torch_sum(torch_log(SrondS))
  elbo = elbo + logSrondS + (1 / 2) * n * q * (torch_log(2*pi) + 1)
  
  # Entropy(q(Z))
  elbo = elbo + torch_nansum(Tau * torch_log(Tau)) # nansum ignores 0 * log(0)
  
  # log p(W | Z) (latent GMM part) 
  nks = Tau$sum(dim=2)
  invLambda = Lambda$inverse() # batch computation on first dim
  for (k in 1:K) {
    if(!torch_equal(Lambda[k,,], Lambda[k,,]$t())) {
      message("Lambda", k, "stop being symm")
    }
    centeredM = M - Mu[k, NULL]
    centeredMk = Tau[k,NULL]$t() * centeredM
    MMtk = centeredMk$t()$mm(centeredM)
    SSk = (Tau[k, NULL]$t() * SrondS)$sum(dim=1)$diag()
    
    elbo = elbo +
      - .5 * nks[k] * Lambda[k,,]$logdet()$real +
      - .5 * invLambda[k,,]$multiply(MMtk + SSk)$sum() # Tr(AB^T) = sum(A * B)
  }
  if(is.nan(as.numeric(elbo))) {
    print(Tau)
    print(Pi)
    stop('?NaN elbo')
  }
  return(elbo)
}


# M-step updates ----------------------------------------------------------

compute_Pi <- function(Tau) {
  Tau$sum(2) / Tau$shape[2]
}


compute_Mu <- function(Tau, M) {
  # Compute the latent GMM mean vectors
  # In function of the variational parameters
  # Returns : a torch_tensor of size (K, q)
  nks = Tau$sum(2)
  # Hacky way to write over K clusters in torch using broadcasting
  Tau[,,NULL]$multiply(M[NULL,,])$divide(nks[,NULL,NULL])$sum(2)
}

compute_Lambda <- function(Tau, M, S, Mu) {
  # Compute the latent GMM covariance matrices
  # In function of the variational parameters
  # Returns : a torch_tensor of size (K, q, q)
  K = Tau$shape[1]
  nks = Tau$sum(2)
  SrondS = torch_multiply(S, S)
  
  lapply(
    1:K,
    function(k) {
      centeredM = M - Mu[k, NULL]
      centeredMk = Tau[k,NULL]$t() * centeredM
      MMtk = centeredMk$t()$mm(centeredM)
      SSk = (Tau[k, NULL]$t() * SrondS)$sum(dim=1)$diag()
      # resize (1,q,q) for torch_cat
      (SSk + MMtk)[NULL,,] / nks[k]
    }) %>% 
    torch_cat()
}

# VE-step updates ---------------------------------------------------------

compute_var_Tau <- function(Y, M, S, C, Lambda, Mu, Pi) {
  # shape (K, n, q)
  centeredM = (M[NULL,,] - Mu[,NULL,])
  # broadcasted vesion of [(m_i - \mu_k)^T \Lambda_k^{-1} (m_i - \mu_k))] over {k,i}
  centeredMLambdaCenteredMt = centeredM[,,NULL,]$matmul(Lambda[,NULL,,]$inverse()$matmul(centeredM[,,,NULL]))$squeeze()
  # broadcasted version of Tr[Lambda_k^{-1} \diag(s_i^2)] over {k, i}
  TrLambdaS = Lambda$diagonal(dim1 = -1, dim2 = -2)[,NULL,]$multiply(S[NULL,])$sum(-1)
  logTau = Pi[,NULL]$log() - .5 * (Lambda$logdet()[,NULL] + centeredMLambdaCenteredMt + TrLambdaS)
  torch_exp(logTau - logTau$logsumexp(dim = 1))
}

compute_var_mean <- function(Y, Tau, C) {
  # no closed form
}

compute_var_S <- function(Y, Tau, M) {
  # no closed form
}

PROFILED_ELBO_MPCA <- function(Y, O, covariates, M, S, Tau, C, Theta) {
  # Same as ELBO_MPCA but the Lambda, Mu & Pi model parameters (latent GMM)
  # Are "profiled out" <=> plugged-in in the ELBO with their expression w.r.t
  # the variational parameters
  
  K = Tau$shape[1]
  n = Tau$shape[2]
  

  Mu = compute_Mu(Tau, M)
  Lambda = compute_Lambda(Tau, M, S, Mu)
  Pi = compute_Pi(Tau)
  ELBO_MPCA(Y, O, covariates, M, S, Tau, C, Theta, Lambda, Mu, Pi)
}

plnmpca_control_init <- function(init_cl = "kmeans", init_C = "pca") {
  return(list(init_cl = init_cl, init_C = init_C))
}



VEM_PLNMPCA <- R6Class("VEM_PLNPCA", 
                      public = list(
                        Y = NULL, 
                        O = NULL,
                        covariates = NULL, 
                        p = NULL, 
                        q = NULL,
                        K = NULL,
                        n = NULL, 
                        d = NULL, 
                        M = NULL, 
                        S = NULL, 
                        Tau = NULL,
                        A = NULL, 
                        C = NULL,
                        Sigma = NULL, 
                        Theta = NULL,
                        Pi = NULL,
                        Mu = NULL,
                        Lambda = NULL,
                        Lambda_vec = NULL, # parametrizatin of Lambda as a vec
                        control_init = NULL,
                        fitted = NULL, 
                        ELBO_list = NULL,
                        profiled = NULL, # do we use a profiled ELBO ?
                        initialize = function(Y,O,covariates,q, K, control_init = plnmpca_control_init(), profiled=TRUE){
                          self$profiled <- profiled
                          self$Y <- Y
                          self$O <- O
                          self$covariates <- covariates
                          self$q = q
                          self$K <- K
                          self$control_init = control_init
                          self$p <- Y$shape[2]
                          self$n <- Y$shape[1]
                          self$d <- covariates$shape[2]
                 
                          ## Model parameters 
                          message('Loadings and Theta initialization ...')
                          if (self$control_init$init_C == "PCA"){
                            self$Theta <- Poisson_reg(Y,O,covariates)$detach()$clone()$requires_grad_(TRUE)
                            self$C <- init_C(Y,O,covariates, self$Theta,q)$detach()$clone()$requires_grad_(FALSE)
                          }
                          else{
                            self$Theta <- torch_zeros(self$d, self$p, requires_grad = TRUE)
                            self$C <- torch_randn(self$p, self$q, requires_grad = FALSE)
                          }
                          message('Cluster initialization ...')
                          cl = switch(self$control_init$init_cl,
                                      "kmeans" = kmeans(scale(Y), centers = K, nstart = 10)$cl,
                                      "random" = max.col(t(rmultinom(n, 1, rep(1/K, K))))
                          )
                          
                          # no grad for Pi params
                          self$Pi = torch_tensor(table(cl) / self$n, requires_grad=FALSE)
                          
                          # Variational parameters q(Z)
                          self$Tau = torch_tensor(as_indicator(cl) %>% t()) # %>% .clip_proba
                          
                          # Mean \mu_k in latent space
                          PSEUDO_COUNT = 1e-6
                          log_mean_per_cluster = (self$Tau[,,NULL]$multiply(self$Y[NULL,,]) + PSEUDO_COUNT)$mean(2)$log()
                          self$Mu = log_mean_per_cluster$matmul(self$C)
                
                          # self$Mu = sapply(Y %>% as.data.frame() %>% split(cl), colMeans) %>% t() %>% torch_tensor()
                          # self$Mu = torch_tensor(logmean_per_cluster %*% self$C, requires_grad = TRUE)

                          # Covar \Lambda_k in latent space
                          self$Lambda = lapply(
                            1:self$K,
                            function(k) {
                              # torch.cov() takes (p,n) tensor as input : t(design)
                              cov_obs_k = (self$Y + PSEUDO_COUNT)$log()$t()$cov(
                                # correction = 0, 
                                aweights = self$Tau[k,]
                                )
                              # return tensor shaped (1,q,q) for torch_cat()
                              self$C$t()$matmul(cov_obs_k)$matmul(self$C)[NULL,,]
                            }
                            ) %>%
                            torch_cat()
                          
                          self$Lambda_vec = lapply(
                            1:self$K,
                            \(k) sdp_matrix_to_reals(self$Lambda[k,,]$squeeze())[NULL,]
                            ) %>%
                            torch_cat()
                          # set grad = TRUE
                          self$C$requires_grad = TRUE
                          self$Tau$requires_grad = TRUE
                          self$Mu$requires_grad = TRUE
                          # self$Lambda$requires_grad = TRUE
                          self$Lambda_vec$requires_grad = TRUE
                          
                          ## Variational parameters
                          self$M <- torch_zeros(self$n, self$q, requires_grad = TRUE)
                          self$S <- torch_ones(self$n, self$q, requires_grad = TRUE)
                          
                          message('Initialization finished.')
                          self$fitted = FALSE
                          self$ELBO_list = c()
                        },
                        fit = function(N_iter, lr, verbose = FALSE){
                          optimizer = optim_rprop(c(self$Theta, self$C, self$M, self$S, self$Tau, self$Mu, self$Lambda), lr = lr) #, self$Pi
                          for (i in 1:N_iter){
                            optimizer$zero_grad()
                            if(self$profiled) {
                              loss = - PROFILED_ELBO_MPCA(self$Y, self$O, self$covariates, self$M, self$S, self$Tau, self$C, self$Theta)
                            } else {
                              loss = - ELBO_MPCA(self$Y, self$O, self$covariates, self$M, self$S, self$Tau, self$C, self$Theta, self$Lambda_vec, self$Mu, self$Pi)
                            }
                            loss$backward()
                            optimizer$step()
                            # normalize Tau without keeping grad
                            with_no_grad({
                              self$Tau <- self$Tau$divide(self$Tau$sum(1, keepdim = TRUE)) %>%
                                .clip_proba()
                              # project onto stiefel with svd ? self$C 
                            })
                            if(!self$profiled) {
                              self$Pi <- self$Tau$sum(2) / self$n
                            } 
                            
                            if(verbose && (i%%1 == 0)){
                              message('i :', as.character(i) )
                              message('ELBO : ', as.character(-loss$item()/(self$n)))
                              pr('MSE C', MSE(self$C - params$C))
                              pr('MSE Theta', MSE(self$Theta- params$Theta))
                            }
                            self$ELBO_list = c(self$ELBO_list, -loss$item()/(self$n))
                          }
                          self$fitted <- TRUE 
                        }, 
                        plot_log_neg_ELBO = function(from = 1){
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
                        },
                        getSigma = function(k=NULL){
                          if(is.null(k)) {
                            return(lapply(
                              1:self$K, 
                              \(k) self$C$mm(self$Lambda[k,,])$mm(self$C$t())[NULL,,])
                            ) %>%
                              torch_cat()
                          } else {
                            return(self$C$mm(self$Lambda[k,,])$mm(self$C$t()))
                          }
                        }
                      ),
                      active = list(
                        getLambda = function(){
                          compute_Lambda(self$Tau, self$M, self$S, self$getMu)
                        },
                        getMu = function(){
                          compute_Mu(self$Tau, self$M)
                        },
                        getPi = function(){
                          compute_Pi(self$Tau)
                        },
                        memberships = function(){
                          self$Tau$argmax(1) %>% as.numeric()
                        }
                      )
)



# Fix seed ----------------------------------------------------------------
seed = as.integer(as.POSIXct(Sys.time()))
seed = 1677852062
set.seed(seed)
torch::torch_manual_seed(seed)

# Fix simulation true parameters ------------------------------------------
n = 100
q = 3
p = 5
K = 2
params = list()
params$Pi = torch_tensor(rep(1/K, K))
params$C = torch_tensor(qr.Q(qr(matrix(rnorm(p*q), p, q))))

params$Mu = torch_zeros(c(K, q))
params$Mu[1,] = torch_ones(q) * 1.5
params$Mu[2,] = - params$Mu[1,]$clone()
params$Mu = torch_tensor(params$Mu)

params$Sigmas = array(0, dim=c(K, q, q))
params$Sigmas[1,,] = 1 * diag(q)
params$Sigmas[2,,] = 0.1 * diag(q)
params$Sigmas = torch_tensor(params$Sigmas)
use_covariates = TRUE 
if (use_covariates) {
  covariates = torch_ones(c(n, 1))
  params$Theta = torch_ones(c(1, 1))
}


n_iter = 1000
profiled_elbo = FALSE

control_mompca = plnmpca_control_init()
simu = simu_plnmpca(n, params, covariates)

km = kmeans(scale(simu$Y), centers=K, nstart=10)
cat('\n ARI kmeans : ', ARI(simu$clusters, km$cl))
plnpca = VEM_PLNMPCA$new(simu$Y, simu$O, covariates, q, K, profiled = profiled_elbo)
plnpca$fit(n_iter,0.1,verbose = TRUE)
plnpca$plot_log_neg_ELBO()



library(aricode)
cat('ARI PLN-MPCA :', ARI(simu$clusters, plnpca$memberships))

library(ggplot2)
for(k in 1:plnpca$K){
  vizmat(as.matrix(plnpca$getSigma(k)))
}
# plnpca$Theta
# true_Theta
# covariates
# 
d = 1L
n = 2000L
p = 2000L
q = 10L






